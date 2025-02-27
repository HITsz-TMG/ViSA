import argparse
import os
print("PID", os.getpid())
os.environ['HF_HOME'] = "/XYFS01/hitsz_bthu_lzy_1/cache"
import sys
sys.path.append("/XYFS01/hitsz_bthu_lzy_1/remote/dense_img/sam2-main")

import random
from tqdm import tqdm
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
import datasets

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

Image.MAX_IMAGE_PIXELS = None
np.random.seed(233)
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

class ImageSegmenter:
    def __init__(self, model_path, config_path, device, batch_size_radio=1):
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        self.sam2 = build_sam2(config_path, model_path, device=device, apply_postprocessing=False)
        self.points_per_batch = 4096
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=16,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=0.4,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            mask_threshold=0.5,
            box_nms_thresh=0.7,
            crop_n_layers=1,
            crop_nms_thresh=0.7,
            crop_overlap_ratio= 512 / 1500 ,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True,
        )
        self.device = device
        self.batch_size_radio = batch_size_radio

    def __call__(self, image):
        image = image.convert("RGB")
        image = smart_resize(image, factor=1, min_pixels=3136, max_pixels=4096 * 28 * 28)
        width, height = image.width, image.height
        image = np.array(image)

        if width * height >= 125000:
            self.mask_generator.points_per_batch = 2048 // self.batch_size_radio
        if width * height >= 250000:
            self.mask_generator.points_per_batch = 1024 // self.batch_size_radio
        if width * height >= 500000:
            self.mask_generator.points_per_batch = 512 // self.batch_size_radio
        if width * height >= 1000000:
            self.mask_generator.points_per_batch = 256 // self.batch_size_radio
        if width * height >= 2000000:
            self.mask_generator.points_per_batch = 128 // self.batch_size_radio
        if width * height >= 4000000:
            self.mask_generator.points_per_batch = 64 // self.batch_size_radio
        if width * height >= 8000000:
            self.mask_generator.points_per_batch = 32 // self.batch_size_radio

        masks = self.mask_generator.generate(image)

        self.mask_generator.points_per_batch = self.points_per_batch

        masks = self.size_filiter(masks, absolute_threshold=0.005)

        return masks

    def size_filiter(self, masks, absolute_threshold=0.0):
        if len(masks) == 0:
            return masks
        segmentation = np.array([mask["segmentation"] for mask in masks])
        index = np.arange(len(segmentation))
        segmentation_size_radio = segmentation.sum(axis=-1).sum(axis=-1) / (segmentation.shape[-1] * segmentation.shape[-2])
        index = index[segmentation_size_radio >= absolute_threshold]
        masks = [masks[x] for x in index]
        return masks


def sam_score_map(instance):
    image_path = os.path.join(args.image_path, instance['image'])
    if "sam score" not in instance or instance["sam score"] == -1:
        try:
            image = Image.open(image_path).convert("RGB")
            masks = model(image=image)
            instance['sam score'] = len(masks)
        except Exception as e:
            print(e)
            instance['sam score'] = -1
    return instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--batch_size_radio', type=int, default=1)
    args = parser.parse_args()

    model = ImageSegmenter(model_path=args.model_path,
                           config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
                           device=torch.device('cuda'),
                           batch_size_radio=args.batch_size_radio
                           )

    chunk_size = args.chunk_size
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    data = datasets.load_from_disk(args.data_path)

    print(f"part total len", len(data))

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
            sam_score_map,
            num_proc=1,
            load_from_cache_file=True
        )
        chunk_data.save_to_disk(chunk_path)

        del chunk_data

