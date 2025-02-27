import os
import requests
from peft import PeftModel
import copy
import datasets
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info, smart_resize
from PIL import Image
from openai import OpenAI
import asyncio
from pprint import pprint
import torch
from io import BytesIO
import base64

try:
    from vllm import LLM, SamplingParams
except:
    pass

def encode_image(image:Image.Image, image_format="PNG") -> str:
    im_file = BytesIO()
    image.save(im_file, format=image_format)
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return im_64

class Qwen2vlAgent:
    MAX_TRY_TIMES = 1

    def __init__(
            self,
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            online_port=None,
            using_hf=False,
            peft_path=None,
            max_new_tokens=128,
            stop=["<|im_end|>"],
            max_model_len=None,
            gpu_memory_utilization=1,
            enforce_eager=False,
            max_num_seqs=16,
            tensor_parallel_size=1,
            num_beams=1,
            temperature=0.0,
            top_p=1.0,
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
            self.online = True
        elif using_hf:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype='auto'
            )
            if peft_path:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    peft_path,
                )
            self.online = False
            self.processor = AutoProcessor.from_pretrained(model_path,
                                                           min_pixels=min_pixels,
                                                           max_pixels=max_pixels
                                                           )
        else:
            self.model = LLM(model_path,
                             gpu_memory_utilization=gpu_memory_utilization,
                             max_model_len=max_model_len,
                             limit_mm_per_prompt={"image": 10},
                             # quantization='awq',
                             dtype='auto',
                             max_num_seqs=max_num_seqs,
                             trust_remote_code=True,
                             enable_prefix_caching=True,
                             enforce_eager=enforce_eager,
                             tensor_parallel_size=tensor_parallel_size,
                             enable_chunked_prefill=enable_chunked_prefill,
                             max_num_batched_tokens=max_num_batched_tokens,
                             )
            self.online = False
            self.processor = AutoProcessor.from_pretrained(model_path,
                                                           min_pixels=min_pixels,
                                                           max_pixels=max_pixels
                                                           )
        self.using_hf = using_hf
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.seed = None
        self.stop = stop

        self.using_io = True

    def format_message(self, message, add_generation_prompt=True):
        prompt = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        image_inputs, video_inputs = process_vision_info(message)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        llm_inputs = {"prompt": prompt}
        if len(mm_data) != 0:
            llm_inputs["multi_modal_data"] = mm_data
        return llm_inputs

    def convert_online_message(self, message, using_io=True):
        return_message = []
        for mid, mes in enumerate(message):
            return_mes = {"role": mes["role"]}
            content = mes['content']
            if mes["role"] == 'user':
                if content[0]['type'] == 'image':
                    assert len(content) == 2 and content[1]['type'] == 'text', str(content)
                    image_path = message[mid]['content'][0]["image"]
                    raw_image = Image.open(image_path).convert('RGB')
                    width, height = raw_image.size
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=28,
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels,
                    )
                    if using_io:
                        image = raw_image.resize((resized_width, resized_height))
                        image_url = f"data:image/png;base64,{encode_image(image, image_format='PNG')}"
                        img_mes = {
                            "type": "image_url",
                            # "image_url": {'url': f"file://{image_path}"},
                            "image_url": {'url': image_url},
                        }
                    else:
                        img_mes = {
                            "type": "image_url",
                            "image_url": {'url': f"file://{image_path}"},
                            "resized_height": resized_height,
                            "resized_width": resized_width,
                        }
                    return_mes['content'] = [
                        img_mes,
                        content[1]
                    ]
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
            best_of=self.num_beams,
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
                    {'answer': o.outputs[0].text,
                     "logits": o.outputs[0].logprobs,
                     "tokens": o.outputs[0].token_ids}
                )

        return response

    def hf_offline_generate(self, messages, batch_size=1):
        response = []
        for b in range(0, len(messages), batch_size):
            conversation = [self.format_message(mes) for mes in messages[b:b + batch_size]]
            text = [conv["prompt"] for conv in conversation]
            image_inputs = [conv["multi_modal_data"]["image"] for conv in conversation]

            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            generated_ids = self.model.generate(
                **inputs,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams
            )
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids):

                output_text = self.processor.decode(
                    out_ids[len(in_ids):], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response.append(
                    {'answer': output_text,
                     "logits": None,
                     "tokens": None}
                )
                print(output_text)

        return response


    def generate(self, messages, batch_size=1):
        if self.online:
            response = asyncio.run(self.online_generate(messages, batch_size=batch_size))
            return response
        elif self.using_hf:
            return self.hf_offline_generate(messages, batch_size=1)
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
        if self.online:
            input_ids = torch.tensor([eval(list(emb.keys())[0]) for emb in embedding])
        else:
            input_ids = torch.tensor([list(emb.keys())[0] for emb in embedding])
        ans_pos = get_ans_pos(input_ids)

        logits = []
        ans_ids = []
        for s, e in ans_pos:
            if self.online:
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
                logprobs = self.post_answer_logits(o.prompt_logprobs)
                response.append(
                    logprobs
                )

        return response

    def embedding(self, messages, batch_size=1):
        if self.online:
            result = asyncio.run(self.online_embedding(messages, batch_size=batch_size))
            return result
        else:
            return self.offline_embedding(messages, batch_size)

