#
# This file is part of the GNU Radio LLM project.
#

import os
import torch

from pathlib import Path

from llm.prompts import build_prompt
from llm.utils import extract_json_from_text

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig


class ModelEngine:
    def __init__(self,
                 model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 fallback_model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct',
                 hf_token_env: str = 'HUGGINGFACE_HUB_TOKEN'):
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.hf_token = os.environ.get(hf_token_env, None)

        self._load_model()

    def _load_model(self):
        if torch.cuda.is_available():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            self.config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type='nf4'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                quantization_config=self.config
            )
        else:
            offload_dir = Path.home() / '.cache' / 'gr_llm_offload'
            offload_dir.mkdir(parents=True, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.fallback_model_name,
                use_fast=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.fallback_model_name,
                device_map='cpu',
                offload_folder=str(offload_dir),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        self.model.config.use_cache = True
        self.model.eval()

    def generate(self, user_prompt: str, max_tokens: int = 1024) -> str:
        prompt = build_prompt(
            self.tokenizer, user_prompt, generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            do_sample=False,
            num_beams=1,
            early_stopping=False,
        )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return extract_json_from_text(decoded)

    def retry_with_feedback(self,
                            user_prompt: str,
                            feedback: str,
                            max_tokens: int = 1024) -> str:
        retry_prompt = (
            f'The previous attempt failed with the following feedback:\n{feedback}\n'
            f'Please try again and produce the correct JSON.\n\n'
            f'Original user prompt: {user_prompt}\n\n'
        )
        return self.generate(retry_prompt, max_tokens=max_tokens)
