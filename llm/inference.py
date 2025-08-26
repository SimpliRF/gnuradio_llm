#
# This file is part of the GNU Radio LLM project.
#

import os
import torch

from typing import Optional
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

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '</s>'
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.model.config.use_cache = True
        self.model.eval()

    def generate(self,
                 user_prompt: str,
                 flowgraph_json: Optional[str] = None,
                 max_tokens: int = 2048) -> str:
        prompt = build_prompt(
            tokenizer=self.tokenizer,
            user_prompt=user_prompt,
            generation_prompt=True,
            flowgraph_json=flowgraph_json
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).to(self.model.device)

        eos_ids = {self.tokenizer.eos_token_id}
        eos_candidates = (
            '<|im_end|>', '</s>', '<|end|>', '<|eot_id|>', '<|endoftext|>'
        )
        for token in eos_candidates:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.add(tid)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=False,
            eos_token_id=list(eos_ids),
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=False,
            output_scores=False,
            temperature=1.0,
            top_p=1.0,
            top_k=None
        )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        results = extract_json_from_text(decoded)
        return results[-1] if results else ''

    def retry_with_feedback(self,
                            user_prompt: str,
                            feedback: str,
                            max_tokens: int = 2048) -> str:
        retry_prompt = (
            f'The previous attempt failed with the following feedback:\n{feedback}\n'
            f'Please try again and produce the correct JSON.\n\n'
            f'Original user prompt: {user_prompt}\n\n'
        )
        return self.generate(retry_prompt, max_tokens=max_tokens)
