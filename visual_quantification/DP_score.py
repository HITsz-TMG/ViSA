import argparse
import sys
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

from prompt_v3 import shot5_prompt, system_message, question_prompt, image_root

def add_message(messages, role='user', image=None, text=None):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if text is not None:
        content.append({"type": "text", "text": text})
    messages.append({"role": role,"content": content})
    return messages



def get_answer(response:str):
    def check(ans, strict=True):
        pattern = r'[1-5]分' if strict else r'[1-5]'
        matches = [(m.group(), m.start()) for m in re.finditer(pattern, ans)]
        return eval(matches[-1][0][0]) if matches else None

    answer = None
    if answer is None and "最终评分" in response: answer = check(response.split("最终评分")[-1])
    if answer is None and "最终评分" in response: answer = check(response.split("最终评分")[-1], strict=False)
    if answer is None and "#" in response: answer = check(response.split("#")[-1])
    if answer is None and "#" in response: answer = check(response.split("#")[-1], strict=False)
    if answer is None and "---" in response: answer = check(response.split("---")[-1])
    if answer is None and "---" in response: answer = check(response.split("---")[-1], strict=False)
    if answer is None: answer = check(response)
    if answer is None: answer = check(response, strict=False)
    return answer

def batch_get_score(instances):
    total_messages = []
    for image_name in instances['image']:
        image_path = os.path.join(
            args.image_path,
            image_name)
        messages = add_message(copy.deepcopy(prompt_messages), role="user", image=str(image_path), text=question_prompt)
        total_messages.append(messages)

    responses = agent.generate(total_messages, batch_size=4)
    instances['agent score'] = []
    instances['answer'] = []
    for x, response in enumerate(responses):
        ans = get_answer(response['answer'])
        if ans is None: ans = -1
        instances['agent score'].append(ans)
        instances['answer'].append(response['answer'])

    return instances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--vllm', type=str, default="Qwen2VL") # Qwen2 InternVL2_5_mpo
    parser.add_argument('--max_retry_times', type=int, default=1)
    args = parser.parse_args()

    prompt_messages = []
    prompt_messages = add_message(prompt_messages, role="system", text=system_message)
    for image_name, answer in shot5_prompt.items():
        image_p = str(os.path.join(image_root, image_name))
        prompt_messages = add_message(prompt_messages, role="user", image=image_p, text=question_prompt)
        prompt_messages = add_message(prompt_messages, role="assistant", text=answer)

    chunk_size = args.chunk_size
    save_path = args.save_path
    MAX_RETRY_TIMES = args.max_retry_times

    os.makedirs(save_path, exist_ok=True)

    data = datasets.load_from_disk(args.data_path)
    chunk_num = int(math.ceil(len(data) / chunk_size)) if chunk_size is not None else 1
    print("chunk_num", chunk_num)

    if args.vllm == "InternVL2_5_mpo":
        agent = InterVL2_5Agent(
            model_path=args.model_path,
            max_num=12,
            max_new_tokens=3000,
            stop=["<|im_end|>", "最终评分\n\n1分", "最终评分\n\n2分", "最终评分\n\n3分", "最终评分\n\n4分", "最终评分\n\n5分"]
        )
    elif args.vllm == "Qwen2VL":
        agent = Qwen2vlAgent(
            model_path=args.model_path,
            max_pixels=1024 * 28 * 28,
            max_new_tokens=3000,
            stop=["<|im_end|>", "最终评分\n\n1分", "最终评分\n\n2分", "最终评分\n\n3分", "最终评分\n\n4分", "最终评分\n\n5分"]
        )
    elif args.vllm == "Llava":
        agent = LlavaOVAgent(
            model_path=args.model_path,
            max_new_tokens=3000,
            stop=["<|im_end|>", "最终评分\n\n1分", "最终评分\n\n2分", "最终评分\n\n3分", "最终评分\n\n4分", "最终评分\n\n5分"],
            dtype='half'
        )
    else:
        raise NotImplementedError

    for chunk_idx in range(chunk_num):
        print("processing chunk", chunk_idx, chunk_num)
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



"""
check mode

CUDA_VISIBLE_DEVICES=3 nohup python /XYFS01/hitsz_bthu_lzy_1/remote/dense_img/Codes/agent_scoring/sft_image_score.py  > /XYFS01/hitsz_bthu_lzy_1/logs/sft_img_score_0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python /XYFS01/hitsz_bthu_lzy_1/remote/dense_img/Codes/agent_scoring/sft_image_score.py  > /XYFS01/hitsz_bthu_lzy_1/logs/sft_img_score_1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python /XYFS01/hitsz_bthu_lzy_1/remote/dense_img/Codes/agent_scoring/sft_image_score.py  > /XYFS01/hitsz_bthu_lzy_1/logs/sft_img_score_2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python /XYFS01/hitsz_bthu_lzy_1/remote/dense_img/Codes/agent_scoring/sft_image_score.py  > /XYFS01/hitsz_bthu_lzy_1/logs/sft_img_score_3.txt 2>&1 &


CUDA_VISIBLE_DEVICES=5 nohup python /XYFS01/hitsz_bthu_lzy_1/remote/dense_img/Codes/agent_scoring/sft_image_score.py  > /XYFS01/hitsz_bthu_lzy_1/logs/sft_img_score_10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python /XYFS01/hitsz_bthu_lzy_1/remote/dense_img/Codes/agent_scoring/sft_image_score.py  > /XYFS01/hitsz_bthu_lzy_1/logs/sft_img_score_11.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python /XYFS01/hitsz_bthu_lzy_1/remote/dense_img/Codes/agent_scoring/sft_image_score.py  > /XYFS01/hitsz_bthu_lzy_1/logs/sft_img_score_12.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python /XYFS01/hitsz_bthu_lzy_1/remote/dense_img/Codes/agent_scoring/sft_image_score.py  > /XYFS01/hitsz_bthu_lzy_1/logs/sft_img_score_13.txt 2>&1 &

"""

