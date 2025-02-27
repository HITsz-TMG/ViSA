import os
import requests
import copy
import datasets
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize
from PIL import Image
from openai import OpenAI
import asyncio
import torchvision.transforms as T
from transformers import AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
from pprint import pprint
import torch
from io import BytesIO
import base64

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models import internvl
except:
    pass

def encode_image(image:Image.Image, image_format="PNG") -> str:
    im_file = BytesIO()
    image.save(im_file, format=image_format)
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return im_64


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InterVL2_5Agent:
    MAX_TRY_TIMES = 1

    def __init__(
            self,
            model_path,
            min_num=1,
            max_num=10,
            online_port=None,
            max_new_tokens=128,
            stop=["<|im_end|>"],
            max_model_len=None,
            gpu_memory_utilization=1,
            enforce_eager=False,
            max_num_seqs=16,
            tensor_parallel_size=1,
            enable_chunked_prefill=False,
            max_num_batched_tokens=None,
    ):
        if online_port:
            self.model_name = model_path
            self.base_url = f"http://localhost:{online_port}/v1"
            self.embeds_url = f"http://localhost:{online_port}/v1/embeddings"
            self.chat_url = f"http://localhost:{online_port}/v1/chat/completions"
            self.gene_url = f"http://localhost:{online_port}/v1/completions"
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=self.base_url,
            )
            self.oneline = True
        else:
            self.model = LLM(model_path,
                             gpu_memory_utilization=gpu_memory_utilization,
                             max_model_len=max_model_len,
                             limit_mm_per_prompt={"image": max_num + 1},
                             quantization='awq' if 'awq' in model_path.lower() else None,
                             dtype='float16',
                             max_num_seqs=max_num_seqs,
                             trust_remote_code=True,
                             enable_prefix_caching=True,
                             enforce_eager=enforce_eager,
                             tensor_parallel_size=tensor_parallel_size,
                             enable_chunked_prefill=enable_chunked_prefill,
                             max_num_batched_tokens=max_num_batched_tokens,
                             )

            self.oneline = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )

        self.min_num = min_num
        self.max_num = max_num
        self.input_size = 448
        self.max_new_tokens = max_new_tokens
        self.temperature = 0.0
        self.top_p = 1.0
        self.num_beams = 1
        self.seed = None
        self.stop = stop

        self.using_io = True

    def format_message(self, message, add_generation_prompt=True):
        image_inputs = []
        new_message = []
        for mes in message:
            new_mes = {'role': mes['role']}
            if mes['role'] == 'user':
                has_img = False
                assert len(mes['content']) <= 2
                for conv in mes['content']:
                    if conv['type'] == 'image':
                        image_inputs.append(conv['image'])
                        has_img = True
                    else:
                        assert conv['type'] == 'text'
                        new_mes['content'] = f"<image>\n{conv['text']}" if has_img else conv['text']
            else:
                new_mes['content'] = mes['content'][0]['text']
            new_message.append(new_mes)

        prompt = self.tokenizer.apply_chat_template(new_message,
                                                    tokenize=False,
                                                    add_generation_prompt=add_generation_prompt)
        mm_data = {}
        if len(image_inputs):
            mm_data["image"] = [Image.open(img).convert('RGB') for img in  image_inputs]
        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        return llm_inputs


    def convert_online_message(self, message, using_io=True):
        return_message = []
        for mid, mes in enumerate(message):
            return_mes = {"role": mes["role"]}
            content = mes['content']
            if mes["role"] == 'user':
                if content[0]['type'] == 'image':
                    assert len(content) == 2 and content[1]['type'] == 'text'
                    image_path = message[mid]['content'][0]["image"]

                    if using_io:
                        raw_image = Image.open(image_path).convert('RGB')
                        images = dynamic_preprocess(raw_image, image_size=self.input_size, use_thumbnail=True, min_num=self.min_num, max_num=self.max_num)
                        return_mes['content'] = [
                            {
                                "type": "image_url",
                                "image_url": {'url': f"data:image/png;base64,{encode_image(img, image_format='PNG')}"}
                            } for img in images
                        ]
                        return_mes['content'].append(content[1])
                    else:
                        raise NotImplementedError
                else:
                    return_mes['content'] = content
            else:
                return_mes['content'] = content
            return_message.append(return_mes)

        return return_message

    async def online_process_message(self, process_fn, semaphore, message, post_process_fn=None):
        async with semaphore:
            online_message = await asyncio.to_thread(self.convert_online_message, message, self.using_io)
            response = await process_fn(online_message)
            if post_process_fn:
                response = await asyncio.to_thread(post_process_fn, response)
            return response

    # -------------------------generate-------------------------

    async def get_generate_response(self, online_message):
        answer = None
        logits = None
        for i in range(self.MAX_TRY_TIMES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_completion_tokens=self.max_new_tokens,
                    messages=online_message,
                    logprobs=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n=self.num_beams,
                    seed=self.seed,
                    stop=self.stop
                )
                answer = response.choices[0].message.content
                tokens = [p.token for p in response.choices[0].logprobs.content]
                logits = [p.logprob for p in response.choices[0].logprobs.content]
                break
            except Exception as e:
                print(e)
                continue
        return {'answer': answer, "logits": logits, "tokens": tokens}

    async def online_generate(self, messages, batch_size=1):
        semaphore = asyncio.Semaphore(batch_size)
        tasks = [self.online_process_message(self.get_generate_response, semaphore, mes) for mes in messages]
        responses = await asyncio.gather(*tasks)
        return responses

    def offline_generate(self, messages, batch_size=1):
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            logprobs=True,
            stop=self.stop,
            seed=self.seed,
            n=self.num_beams,
        )
        response = []
        for b in range(0, len(messages), batch_size):
            conversation = [self.format_message(mes) for mes in messages[b:b + batch_size]]
            outputs = self.model.generate(conversation,
                                          sampling_params=sampling_params,
                                          use_tqdm=True)
            for o in outputs:
                response.append(
                    {'answer': o.outputs[0].text,
                     "logits": o.outputs[0].logprobs,
                     "tokens": o.outputs[0].token_ids}
                )

        return response

    def generate(self, messages, batch_size=1):
        if self.oneline:
            response = asyncio.run(self.online_generate(messages, batch_size=batch_size))
            return response
        else:
            return self.offline_generate(messages, batch_size)

    # -------------------------embedding-------------------------

    def post_answer_logits(self, embedding):

        def get_ans_pos(input_ids):
            start_value = 151644
            end_value = 151645
            sequence_to_find = torch.tensor([151644, 77091, 198])

            start_indices = (input_ids == start_value).nonzero(as_tuple=True)[0]

            results = []
            for start_index in start_indices:
                if start_index + len(sequence_to_find) < len(input_ids):
                    if torch.equal(input_ids[start_index:start_index + len(sequence_to_find)], sequence_to_find):
                        ans_start_index = start_index.item() + len(sequence_to_find)
                        ans_len = (input_ids[ans_start_index:] == end_value).nonzero(as_tuple=True)[0][0].item()
                        results.append((ans_start_index, ans_start_index + ans_len))

            return results

        embedding = embedding[1:] # ignore first None value
        assert all(len(emb) == 1 for emb in embedding)
        if self.oneline:
            input_ids = torch.tensor([eval(list(emb.keys())[0]) for emb in embedding])
        else:
            input_ids = torch.tensor([list(emb.keys())[0] for emb in embedding])
        ans_pos = get_ans_pos(input_ids)

        logits = []
        ans_ids = []
        for s, e in ans_pos:
            if self.oneline:
                logit = [list(emb.values())[0]['logprob'] for emb in embedding[s:e]]
            else:
                logit = [list(emb.values())[0].logprob for emb in embedding[s:e]]
            ans_id = input_ids[s:e].tolist()
            logits.append(logit)
            ans_ids.append(ans_id)

        return {'ans_ids': ans_ids, 'logits': logits}


    async def get_embedding_response(self, online_message):
        logprobs = None
        for i in range(self.MAX_TRY_TIMES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_completion_tokens=0,
                    max_tokens=1,
                    messages=online_message,
                    logprobs=False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n=self.num_beams,
                    seed=self.seed,
                    extra_body={
                        "prompt_logprobs": 0,
                        "add_generation_prompt": False
                    },
                )
                logprobs = response.prompt_logprobs
                break
            except Exception as e:
                print(e)
                continue
        return logprobs

    async def online_embedding(self, messages, batch_size=1):
        semaphore = asyncio.Semaphore(batch_size)
        tasks = [self.online_process_message(self.get_embedding_response, semaphore, mes, self.post_answer_logits) for mes in messages]
        logits = await asyncio.gather(*tasks)
        return logits

    def offline_embedding(self, messages, batch_size=1):
        sampling_params = SamplingParams(
            logprobs=1,
            prompt_logprobs=0,
            max_tokens=1,
        )
        response = []
        for b in range(0, len(messages), batch_size):
            conversation = [self.format_message(mes) for mes in messages[b:b + batch_size]]
            outputs = self.model.generate(
                conversation,
                sampling_params=sampling_params,
                use_tqdm=True
            )
            for o in outputs:
                response.append(
                    self.post_answer_logits(o.prompt_logprobs)
                )

        return response

    def embedding(self, messages, batch_size=1):
        if self.oneline:
            result = asyncio.run(self.online_embedding(messages, batch_size=batch_size))
            return result
        else:
            return self.offline_embedding(messages, batch_size)

