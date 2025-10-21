
import os
import time
import base64
import asyncio
from PIL import Image
from io import BytesIO
from typing import Union
from httpx import Timeout
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI

from config import VLLM_MODEL_CONFIG

class LVLM:
    def __init__(self, model_id:str="lvlm"):
        self.model_id = model_id
        return 
    
    def visual_chat(self, images:Union[Image.Image, list[Image.Image]], questions:Union[str, list[str]], **chat_args):
        return [""] * len(questions)
    
    def chat(self, questions:Union[str, list[str]], **chat_args):
        return [""] * len(questions)


class API_LVLM(LVLM):
    def __init__(self, model_id:str="gpt-4o-mini", api_key:str=None, base_url:str=None, enable_async=True):
        if api_key is None:
            api_key = os.getenv("API_KEY")
        if base_url is None:
            base_url = os.getenv("BASE_URL")

        self.enable_async = enable_async
        self.interval = 0.5
        self.timeout = Timeout(60.0)
        
        if model_id.startswith("azure-") or "azure" in base_url:
            self.model_id = model_id.removeprefix("azure-")
            if enable_async:
                self.client = AsyncAzureOpenAI(azure_endpoint=base_url, api_key=api_key, api_version="2024-12-01-preview", timeout=self.timeout)
            else:
                self.client = AzureOpenAI(azure_endpoint=base_url, api_key=api_key, api_version="2024-12-01-preview", timeout=self.timeout)
        else:
            self.model_id = model_id
            if enable_async:
                self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=self.timeout)
            else:
                self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=self.timeout)

    @staticmethod
    def encode_image(image:Image.Image):
        assert isinstance(image, Image.Image)

        image_file = BytesIO()
        if image.mode == "RGBA":
            image.save(image_file, format="PNG")
            ext_name = "png"
        else:
            image.save(image_file, format="JPEG")
            ext_name = "jpeg"
        base64_image = base64.b64encode(image_file.getvalue()).decode('utf-8')
        
        return ext_name, base64_image
    
    async def _stream_chat_once(self, messages: list[dict], max_new_tokens: int, temperature: float, max_retries: int = 3, retry_delay: float = 2.0) -> str:
        attempt = 0
        while attempt < max_retries:
            try:
                stream_response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_completion_tokens=max_new_tokens,
                    temperature=temperature,
                    stream=True
                )

                partial_text = []
                async for chunk in stream_response:
                    if chunk.choices[0].delta.content is not None:
                        partial_text.append(chunk.choices[0].delta.content)
                
                return "".join(partial_text)
            except Exception as e:
                attempt += 1
                if "Error code: 429" in str(e) and "retry after" in str(e) and attempt < 3:
                    try:
                        wait_time = float(str(e).split("retry after")[1].split("second")[0].strip())
                        await asyncio.sleep(wait_time)
                    except:
                        pass
                    continue
                
                if attempt >= max_retries:
                    return f"Request failed after {max_retries} attempts: {str(e)}"
                await asyncio.sleep(retry_delay)

    async def _async_batch_infer(self, messages_batch: list[list[dict]], max_new_tokens: int, temperature: float, max_retries: int = 3, retry_delay: float = 2.0) -> list[str]:
        tasks = []
        for messages in messages_batch:
            tasks.append(
                asyncio.create_task(
                    self._stream_chat_once(messages=messages, max_new_tokens=max_new_tokens, temperature=temperature, max_retries=max_retries, retry_delay=retry_delay)
                )
            )
        batch_responses = await asyncio.gather(*tasks)
        return batch_responses
    
    def _batch_infer(self, messages_batch: list[list[dict]], max_new_tokens: int, temperature: float, max_retries: int = 3, retry_delay: float = 2.0) -> list[str]:    
        responses = []
        for messages in messages_batch:
            attempt = 0
            while attempt < max_retries:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        max_completion_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    responses.append(response.choices[0].message.content)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= max_retries:
                        responses.append(f"Request failed after {max_retries} attempts: {str(e)}")
                        break
                    time.sleep(retry_delay)
        return responses

    def visual_chat(self, images:Union[Image.Image, list[Image.Image]], questions:Union[str, list[str]], system_messages:list=None, max_new_tokens:int=None, temperature:float=0, max_retries:int=3, retry_delay:float=2.0)->list[str]:
        questions_ = questions
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(questions, str):
            questions = [questions]
        assert len(images) == len(questions), f"Length of images and questions should be the same, but got {len(images)} and {len(questions)}."

        base64_images = [self.encode_image(image) for image in images]

        messages_batch = [[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/{ext_name};base64,{base64_image}"}}
            ]
        }] for question, (ext_name, base64_image) in zip(questions, base64_images)]

        if system_messages is not None:
            assert isinstance(system_messages, list) and isinstance(system_messages[0], dict)
            messages_batch = [system_messages + messages for messages in messages_batch]

        if self.enable_async:
            responses = asyncio.run(self._async_batch_infer(messages_batch, max_new_tokens, temperature, max_retries, retry_delay))
        else:
            responses = self._batch_infer(messages_batch, max_new_tokens, temperature, max_retries, retry_delay)
        
        if isinstance(questions_, str):
            responses = responses[0]
        
        return responses

    def chat(self, questions: Union[str, list[str]], system_messages:str=None, max_new_tokens: int = None, temperature: float = 0, max_retries: int = 3, retry_delay: float = 2.0) -> list[str]:
        questions_ = questions
        if isinstance(questions, str):
            questions = [questions]

        messages_batch = [[{
            "role": "user",
            "content": question
        }] for question in questions]

        if system_messages is not None:
            assert isinstance(system_messages, list) and isinstance(system_messages[0], dict)
            messages_batch = [system_messages + messages for messages in messages_batch]
        
        if self.enable_async:
            responses = asyncio.run(self._async_batch_infer(messages_batch, max_new_tokens, temperature, max_retries, retry_delay))
        else:
            responses = self._batch_infer(messages_batch, max_new_tokens, temperature, max_retries, retry_delay)
        
        if isinstance(questions_, str):
            responses = responses[0]
        
        return responses
    
    def msg_chat(self, messages:Union[list[dict], list[list[dict]]], max_new_tokens:int=None, temperature:float=0.0, max_retries:int=3, retry_delay:float=2.0)->str:
        messages_ = messages
        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        assert isinstance(messages, list) and isinstance(messages[0], list) and isinstance(messages[0][0], dict)

        # check images in messages, encode image to base64 if necessary
        for msgs in messages:
            for msg in msgs:
                for content in msg["content"]:
                    if content["type"] == "image":
                        assert isinstance(content["image"], Image.Image)
                        ext_name, base64_image = self.encode_image(content["image"])
                        msg["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{ext_name};base64,{base64_image}"}
                        })
                        msg["content"].remove(content)

        if self.enable_async:
            responses = asyncio.run(self._async_batch_infer(messages, max_new_tokens, temperature, max_retries, retry_delay))
        else:
            responses = self._batch_infer(messages, max_new_tokens, temperature, max_retries, retry_delay)
        
        if len(messages_) == 1:
            responses = responses[0]
        
        return responses

    def list_models(self):
        model_list = self.client.models.list()
        async def _inner_list():
            res_model_list = []
            async for model in model_list:
                res_model_list.append((model.id, model.owned_by))
            return res_model_list
        res_model_list = asyncio.run(_inner_list())
        return res_model_list


class VLLM_LVLM(LVLM):
    def __init__(self, model_id:str, tensor_parallel_size:int):
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size

        model_id_ = model_id.split("/")[-1]
        if model_id_ not in VLLM_MODEL_CONFIG:
            model_id_ = "Qwen2-VL-7B-Instruct" # for training checkpoint
        
        self.prompt_template = VLLM_MODEL_CONFIG[model_id_]["prompt_template"]
        self.visual_prompt_template = VLLM_MODEL_CONFIG[model_id_]["visual_prompt_template"]
        self.templates = VLLM_MODEL_CONFIG[model_id_]["templates"]
        self.stop_token_ids = VLLM_MODEL_CONFIG[model_id_]["stop_token_ids"]
        
        self.lora_requests = None
        if "Phi-4-multimodal-instruct" in model_id:
            self.lora_requests = LoRARequest("vision", 1, os.path.join(model_id, "vision-lora"))
        
        self.llm = LLM(model=self.model_id, trust_remote_code=True, tensor_parallel_size=self.tensor_parallel_size, **VLLM_MODEL_CONFIG[model_id_]["llm_args"])

    def chat(self, questions:Union[str, list[str]], max_new_tokens:int=None, temperature:float=0.0):
        sampling_params = SamplingParams(temperature=temperature, stop_token_ids=self.stop_token_ids, max_tokens=max_new_tokens)

        questions_ = questions
        if isinstance(questions, str):
            questions = [questions]

        inputs = [self.prompt_template.format(question) for question in questions]

        outputs = self.llm.generate(inputs, sampling_params, use_tqdm=False)
        generated_texts = [output.outputs[0].text.strip() for output in outputs]

        if isinstance(questions_, str):
            generated_texts = generated_texts[0]

        return generated_texts
    
    def visual_chat(self, images:Union[Image.Image, list[Image.Image]], questions:Union[str, list[str]], max_new_tokens:int=None, temperature:float=0.0):
        # assume one question is associated with one image
        questions_ = questions
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(questions, str):
            questions = [questions]
        assert len(images) == len(questions)

        sampling_params = SamplingParams(temperature=temperature, stop_token_ids=self.stop_token_ids, max_tokens=max_new_tokens)

        inputs = [{
            "prompt": self.visual_prompt_template.format(question),
            "multi_modal_data": {
                "image": image
            }
        } for image, question in zip(images, questions)]

        outputs = self.llm.generate(inputs, sampling_params, use_tqdm=False, lora_request=self.lora_requests)
        generated_texts = [output.outputs[0].text.strip() for output in outputs]

        if isinstance(questions_, str):
            generated_texts = generated_texts[0]

        return generated_texts
    
    def parse_messages(self, messages: list[list[dict]]) -> tuple[list[Image.Image], list[str]]:
        images = []
        prompts = []
        for msgs in messages:
            prompt = self.templates["system"]
            for msg in msgs:
                assert msg["role"] in ["user", "assistant"] and isinstance(msg["content"], list)
                text, image = None, None
                for content in msg["content"]:
                    if content["type"] == "text" and text is None:
                        text = content["text"]
                    if content["type"] == "image" and image is None:
                        image = content["image"]
                
                if image is not None:
                    text = self.templates["image"] + text
                    images.append(image)
                else:
                    text = self.templates["wo_image"] + text
                
                if msg["role"] == "user":
                    prompt += self.templates["user"].format(text)
                elif msg["role"] == "assistant":
                    prompt += self.templates["assistant"].format(text)

            prompt += self.templates["generate"]
            prompts.append(prompt)
        assert len(images) == len(prompts)
        return images, prompts
    
    def msg_chat(self, messages:Union[list[dict], list[list[dict]]], max_new_tokens:int=None, temperature:float=0.0) -> str:
        messages_ = messages
        if isinstance(messages, list) and isinstance(messages[0], dict):
            messages = [messages]
        assert isinstance(messages, list) and isinstance(messages[0], list) and isinstance(messages[0][0], dict)

        images, prompts = self.parse_messages(messages)
        
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            }
        } for image, prompt in zip(images, prompts)]

        sampling_params = SamplingParams(temperature=temperature, stop_token_ids=self.stop_token_ids, max_tokens=max_new_tokens)

        outputs = self.llm.generate(inputs, sampling_params, use_tqdm=False)
        generated_texts = [output.outputs[0].text.strip() for output in outputs]

        if len(messages_) == 1:
            generated_texts = generated_texts[0]

        return generated_texts

