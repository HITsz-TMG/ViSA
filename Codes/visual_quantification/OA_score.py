import argparse
import os
import sys
import copy
import random
from tqdm import tqdm
import time
from typing import List
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
import datasets
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoProcessor, AutoTokenizer, AutoModelForZeroShotObjectDetection, GroundingDinoProcessor
from transformers.image_transforms import center_to_corners_format
from transformers.utils import TensorType
import math

Image.MAX_IMAGE_PIXELS = None

global_output_kwargs = {
    'text_kwargs': {'add_special_tokens': True, 'padding': True, 'stride': 0, 'return_overflowing_tokens': False,
                    'return_special_tokens_mask': False, 'return_offsets_mapping': False,
                    'return_token_type_ids': True, 'return_length': False, 'verbose': True, 'truncation': True,
                    'max_length': 255, 'return_tensors': 'pt'}, 'images_kwargs': {'return_tensors': 'pt'},
    'audio_kwargs': {'return_tensors': 'pt'}, 'videos_kwargs': {'return_tensors': 'pt'},
    'common_kwargs': {'return_tensors': 'pt'}}

NUM_PROC = 16


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(image, factor: int, min_pixels: int, max_pixels: int):
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    width, height = image.size
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    image = image.resize((w_bar, h_bar))
    return image


def get_tag_len(instance, index):
    assert INPUT_PROMPT.replace('{}', '') not in instance['tag']
    input_ids = tokenizer(INPUT_PROMPT.format(instance['tag']), add_special_tokens=False, padding=False, truncation=False).input_ids
    instance['tag_len'] = len(input_ids)
    instance['id'] = index
    return instance


def concanate_tag(data_path, max_len=None, repeat_num=1):
    print('start processing tag files')
    with open(data_path) as f:
        lines = f.readlines()
    tag_data = datasets.Dataset.from_dict({"tag": [l.strip() for l in lines]})
    tag_data = tag_data.map(get_tag_len,
                    batched=False,
                    with_rank=False,
                    with_indices=True,
                    num_proc=NUM_PROC
                    )
    print('start processing group tag files', f"tag data len: {len(tag_data)}")
    ori_len = len(tag_data)
    if max_len is not None:
        tag_data = tag_data.filter(lambda instance: instance['tag_len'] < max_len, num_proc = NUM_PROC)
        print(f"filtered data num : {len(tag_data) - ori_len}")

    group_tag_data = []
    group_tag_id = []
    batch_ids = []
    for ren in range(repeat_num):
        tag_data = tag_data.shuffle(seed=233 + ren)
        cur_group = [tag_data[0]['tag']]
        cur_ids = [tag_data[0]['id']]
        cur_len = tag_data[0]['tag_len']
        ids2group = []
        for i in range(1, len(tag_data)):
            if max_len is None or cur_len + tag_data[i]['tag_len'] > max_len:
                group_tag_data.append(copy.deepcopy(cur_group))
                group_tag_id.append(copy.deepcopy(cur_ids))
                batch_ids.append(ren)
                cur_group = [tag_data[i]['tag']]
                cur_ids = [tag_data[i]['id']]
                cur_len = tag_data[i]['tag_len']
            else:
                cur_group.append(tag_data[i]['tag'])
                cur_ids.append(tag_data[i]['id'])
                cur_len += tag_data[i]['tag_len']
            ids2group.append(len(group_tag_data))
        group_tag_data.append(copy.deepcopy(cur_group))
        group_tag_id.append(copy.deepcopy(cur_ids))
        batch_ids.append(ren)
        print(len(group_tag_data))

    group_tag_dataset = {'batch_ids': batch_ids, 'tags': group_tag_data, 'tags_id': group_tag_id, 'id':list(range(len(group_tag_data)))}
    group_tag_dataset = datasets.Dataset.from_dict(group_tag_dataset)

    return group_tag_dataset


def split_tag2index(input_ids):
    left_idx = 1 # bos
    right_idx = input_ids.shape[-1] - 1 # eos

    start = left_idx
    indices = []
    for i in range(left_idx, right_idx):
        if input_ids[i] == 1012:
            indices.append((start, i))
            start = i + 1

    return indices

def post_process_grounded_object_detection(
    processor,
    logits,
    boxes,
    input_ids,
    tag_split_index,
    tag_batch_ids,
    tag_score_threshold: float = 0.25,
    target_sizes: Union[TensorType, List[Tuple]] = None,
    box_size_max_threshold=1.0
):
    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    probs = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
    scores = torch.max(probs, dim=-1)[0]  # (batch_size, num_queries)

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(boxes)
    # Convert from relative [0, 1] to absolute [0, height] coordinates
    if target_sizes is not None:
        if isinstance(target_sizes, List):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        original_box_size = (boxes[:,:,2] - boxes[:,:,0]) * ((boxes[:,:,3] - boxes[:,:,1]))
        boxes = boxes * scale_fct[:, None, :]

    score_results = []
    box_results = []
    label_results = []
    batch_ids_result = []
    for idx, (s, b, p, ob, index, bid) in enumerate(zip(scores, boxes, probs, original_box_size, tag_split_index, tag_batch_ids)):
        box = b[s > tag_score_threshold]
        if len(box) == 0: continue
        prob = p[s > tag_score_threshold]
        ob = ob[s > tag_score_threshold]

        box = box[ob < box_size_max_threshold]
        if len(box) == 0: continue
        prob = prob[ob < box_size_max_threshold]

        for st,ed in index:
            tag_prob = prob[:, st:ed].mean(dim=-1)
            tag_index = tag_prob > tag_score_threshold
            tag_prob = tag_prob[tag_index]
            if len(tag_prob):
                tag_box = box[tag_index]
                score_results += tag_prob.tolist()
                box_results += tag_box.tolist()
                label_results += [processor.decode(input_ids[idx][st:ed])] * len(tag_prob)
                batch_ids_result.append(bid)

    return score_results, label_results, box_results, batch_ids_result


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2

    inter_x1 = max(x1, x1_b)
    inter_y1 = max(y1, y1_b)
    inter_x2 = min(x2, x2_b)
    inter_y2 = min(y2, y2_b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_b - x1_b) * (y2_b - y1_b)

    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def batch_tokenize_tags_data(tag_data, processor, tag_batch_size=1):
    global global_output_kwargs

    def format_tags(tags):
        return ''.join([INPUT_PROMPT.format(t) for t in tags])

    batch_tag_encoding = []
    batch_tag_split_index = []
    batch_ids = []
    for i in range(0, len(tag_data), tag_batch_size):
        batch_size = min(len(tag_data) - i, tag_batch_size)
        batch_tags = [format_tags(tag_data[j]['tags']) for j in range(i, min(len(tag_data), i + tag_batch_size))]
        tag_encoding = processor.tokenizer(
            text=batch_tags,
            **global_output_kwargs["text_kwargs"],
        )
        tag_split_index = [split_tag2index(tag_encoding.input_ids[j]) for j in range(len(tag_encoding.input_ids))]
        batch_tag_encoding.append(tag_encoding)
        batch_tag_split_index.append(tag_split_index)
        batch_ids.append([tag_data[j]['batch_ids'] for j in range(i, min(len(tag_data), i + tag_batch_size))])

    return batch_tag_encoding, batch_tag_split_index, batch_ids


def get_tag_single(image, batch_tag_encoding, batch_tag_split_index, batch_ids_list, model, processor, dep_iou=0.9):
    global global_output_kwargs
    image_results = {"scores": [], "labels": [], "boxes": [], "batch_belong": []}

    max_batch_size = max([inputs["input_ids"].shape[0] for inputs in batch_tag_encoding])
    encoding_image_processor = processor.image_processor(image, **global_output_kwargs["images_kwargs"])
    pixel_values = encoding_image_processor["pixel_values"].expand(max_batch_size, -1, -1, -1).to(model.device)
    pixel_mask = encoding_image_processor["pixel_mask"].expand(max_batch_size, -1, -1).to(model.device)

    for batch_ids, tag_encoding in enumerate(batch_tag_encoding):

        inputs = {k: v.to(model.device) for k,v in tag_encoding.items()}
        inputs.update(
            {
                "pixel_values": pixel_values[:inputs["input_ids"].shape[0]],
                "pixel_mask": pixel_mask[:inputs["input_ids"].shape[0]],
            }
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu()
        pred_boxes = outputs.pred_boxes.cpu()
        del outputs

        input_ids = inputs["input_ids"].cpu()
        del inputs

        scores, labels, boxes, batch_belong = post_process_grounded_object_detection(
            processor=processor,
            logits=logits,
            boxes=pred_boxes,
            input_ids=input_ids,
            tag_split_index=batch_tag_split_index[batch_ids],
            tag_batch_ids=batch_ids_list[batch_ids],
            tag_score_threshold=TAG_THRESHOLD,
            target_sizes=[image.size[::-1]]  * input_ids.shape[0],
            box_size_max_threshold=BOX_SIZE_THRESHOLD
        )

        if len(labels):
            image_results["scores"] += scores
            image_results["labels"] += labels
            image_results["boxes"] += boxes
            image_results["batch_belong"] += boxes

    if dep_iou is not None:
        gid = 0
        label2group = {}
        group_score = []
        group_label = []
        group_box = []
        group_batchid = []
        for s, l, b, bid in zip(image_results["scores"], image_results["labels"], image_results["boxes"],
                                image_results["batch_belong"]):
            if l not in label2group:
                label2group[l] = gid
                group_score.append([s])
                group_box.append([b])
                group_label.append([l])
                group_batchid.append([bid])
                gid += 1
            else:
                cur_gid = label2group[l]
                group_score[cur_gid].append(s)
                group_box[cur_gid].append(b)
                group_label[cur_gid].append(l)
                group_batchid[cur_gid].append(bid)

        final_score_results = []
        final_label_results = []
        final_box_results = []
        for group_s, group_b, group_l, group_bid in zip(group_score, group_box, group_label, group_batchid):
            grouped = list(zip(group_s, group_b, group_l, group_bid))
            sorted_grouped = sorted(grouped, key=lambda x: x[0])
            sorted_group_s, sorted_group_b, sorted_group_l, sorted_group_bid = zip(*sorted_grouped)
            for i in range(len(sorted_group_s)):
                flag = True
                for j in range(i + 1, len(sorted_group_s)):
                    if sorted_group_bid[j] != sorted_group_bid[i] and compute_iou(sorted_group_b[i],
                                                                                  sorted_group_b[j]) > 0.9:
                        flag = False
                        break
                if flag:
                    final_score_results.append(sorted_group_s[i])
                    final_label_results.append(sorted_group_l[i])
                    final_box_results.append(sorted_group_b[i])
        image_results = {"scores": final_score_results, "labels": final_label_results, "boxes": final_box_results}


    return image_results

def dino_score_map(instance):
    global batch_tag_dataloader, batch_tag_split_index, batch_ids_list, model, processor, DEP_IOU
    if instance.get("dino_valid"): return instance
    image_path = os.path.join(args.image_path, instance['image'])
    try:
        image = Image.open(image_path).convert("RGB")
        image = smart_resize(image, factor=1, min_pixels=3136, max_pixels=4096 * 28 * 28)
        image_result = get_tag_single(image, batch_tag_dataloader, batch_tag_split_index, batch_ids_list, model, processor, dep_iou=DEP_IOU)
        instance["dino_scores"] = image_result['scores']
        instance["dino_labels"] = image_result['labels']
        instance["dino_boxes"] = image_result['boxes']
        instance['dino_valid'] = True
    except Exception as e:
        print(e)
        instance["dino_scores"] = []
        instance["dino_labels"] = []
        instance["dino_boxes"] = []
        instance['dino_valid'] = False

    return instance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tag_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--chunk_size', type=int, default=None)
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_PATH).cuda()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    TAG_PATH = args.tag_path
    TAG_BATCH_SIZE = 30
    GROUP_TAG_MAXLEN = 250
    REPEAT_TAG_NUM = 3
    INPUT_PROMPT = "{}. "
    group_tag_data = concanate_tag(TAG_PATH, max_len=GROUP_TAG_MAXLEN, repeat_num=REPEAT_TAG_NUM)
    batch_tag_dataloader, batch_tag_split_index, batch_ids_list = \
        batch_tokenize_tags_data(tag_data=group_tag_data,
                                 processor=processor,
                                 tag_batch_size=TAG_BATCH_SIZE)
    print("len(group_tag_data)", len(group_tag_data))

    TAG_THRESHOLD = 0.4
    DEP_IOU = 0.95
    BOX_SIZE_THRESHOLD = 0.95

    chunk_size = args.chunk_size
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    data = datasets.load_from_disk(args.data_path)

    torch.inference_mode().__enter__()
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    chunk_num = int(math.ceil(len(data) / chunk_size)) if chunk_size is not None else 1
    for chunk_idx in range(chunk_num):
        chunk_path = os.path.join(save_path, f"{chunk_idx}")
        print(f'start processing chunk: {chunk_idx}')

        chunk_data = data.shard(num_shards=chunk_num, index=chunk_idx)

        if os.path.exists(chunk_path) and os.path.exists(os.path.join(chunk_path, "dataset_info.json")):
            continue

        chunk_data = chunk_data.map(
            dino_score_map,
            num_proc=1,
            load_from_cache_file=True
        )
        chunk_data.save_to_disk(chunk_path)

