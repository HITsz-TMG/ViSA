import sys
import argparse
import os
import copy
import math
import time
import datasets
import json
import re

from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from ..vllms.VllmInternVL2_5 import InterVL2_5Agent
except:
    from vllms.VllmInternVL2_5 import InterVL2_5Agent

try:
    from ..vllms.VllmQwen2VL import Qwen2vlAgent
except:
    from vllms.VllmQwen2VL import Qwen2vlAgent

try:
    from ..vllms.VllmLlavaov import LlavaOVAgent
except:
    from vllms.VllmLlavaov import LlavaOVAgent


def add_message(messages, role='user', image=None, text=None):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if text is not None:
        content.append({"type": "text", "text": text})
    messages.append({"role": role,"content": content})
    return messages


def batch_get_score(instances):

    def compute_cross_entropy_loss(log_probs):
        N = len(log_probs)
        ce_loss = sum(log_probs) / N
        return - ce_loss

    def compute_conditional_mutual_information(log_probs_x, log_probs_y):
        ce_loss_x = compute_cross_entropy_loss(log_probs_x)
        ce_loss_xy = compute_cross_entropy_loss(log_probs_y)
        conditional_mi = ce_loss_x - ce_loss_xy
        return conditional_mi

    total_messages = []
    total_noimg_messages = []
    for conversations, image_name in zip(instances['conversations'], instances['image']):
        img_path = os.path.join(
            args.image_path,
            image_name)

        messages = []
        noimg_messages = []
        for x, conv in enumerate(conversations):
            if x % 2 == 0:
                assert conv['from'] == 'human', conversations
                messages = add_message(messages, role="user", image=str(img_path) if x == 0 else None, text=conv['value'])
                noimg_messages = add_message(noimg_messages, role="user", image=None, text=conv['value'])
            else:
                assert conv['from'] == 'gpt', conversations
                messages = add_message(messages, role="assistant", image=None, text=conv['value'])
                noimg_messages = add_message(noimg_messages, role="assistant", image=None, text=conv['value'])
        total_messages.append(messages)
        total_noimg_messages.append(noimg_messages)

    img_response_len = len(total_messages)
    responses = agent.embedding(total_messages + total_noimg_messages, batch_size=len(total_messages) + len(total_noimg_messages))

    instances['image_ans_ids'] = []
    instances['noimage_ans_ids'] = []
    instances['image_logits'] = []
    instances['noimage_logits'] = []
    for x, response in enumerate(responses[:img_response_len]):
        instances['image_ans_ids'].append(response["ans_ids"])
        instances['image_logits'].append(response["logits"])
    for x, response in enumerate(responses[img_response_len:]):
        instances['noimage_ans_ids'].append(response["ans_ids"])
        instances['noimage_logits'].append(response["logits"])

    instances["PT score"] = []
    instances["IM score"] = []
    for ins_id, instance in enumerate(instances):
        image_logits = []
        noimage_logits = []
        for img_logit, noimg_logit in zip(instance["image_logits"], instance["noimage_logits"]):
            image_logits += img_logit
            noimage_logits += noimg_logit
        instances["IM score"].append(
            compute_conditional_mutual_information(noimage_logits, image_logits)
        )

        image_logits = []
        for img_logit in instance["image_logits"]:
            if args.prior_token_num != None:
                image_logits += img_logit[:args.prior_token_num]
            else:
                image_logits += img_logit
        instances["PT score"].append(
            compute_cross_entropy_loss(image_logits)
        )

    return instances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--prior_token_num', type=int, default=None)
    parser.add_argument('--vllm', type=str, default="Qwen2VL")
    args = parser.parse_args()

    chunk_size = args.chunk_size
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    data = datasets.load_from_disk(args.data_path)
    chunk_num = int(math.ceil(len(data) / chunk_size)) if chunk_size is not None else 1
    print("chunk_num", chunk_num)


    if args.vllm == "InternVL2_5_mpo":
        agent = InterVL2_5Agent(
            model_path=args.model_path,
            max_num=12,
            max_new_tokens=3000,
        )
    elif args.vllm == "Qwen2VL":
        agent = Qwen2vlAgent(
            model_path=args.model_path,
            max_pixels=1024 * 28 * 28,
            max_new_tokens=3000,
        )
    elif args.vllm == "Llava":
        agent = LlavaOVAgent(
            model_path=args.model_path,
            max_new_tokens=3000,
            dtype='half'
        )
    else:
        raise NotImplementedError

    for chunk_idx in range(chunk_num):
        print(f"processing chunk: {chunk_idx} / {chunk_num}")
        chunk_path = os.path.join(save_path, f"{chunk_idx}")
        if os.path.exists(chunk_path) and os.path.exists(os.path.join(chunk_path, "dataset_info.json")):
            continue
        chunk_data = data.shard(num_shards=chunk_num, index=chunk_idx)

        chunk_data = chunk_data.map(
            batch_get_score,
            num_proc=1,
            batched=True,
            batch_size=4,
        )
        chunk_data.save_to_disk(chunk_path)

