#
# This file is part of the GNU Radio LLM project.
#

import torch

from llm.prompts import get_system_prompt
from llm.utils import extract_json_from_text

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import TextStreamer
from transformers.utils.quantization_config import BitsAndBytesConfig


class ModelEngine:
    def __init__(self, model_name: str = 'mistral-7b-instruct'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=self.config
        )
        self.streamer = TextStreamer(self.tokenizer)

    def generate(self, user_prompt: str, max_tokens: int = 1024) -> str:
        prompt = get_system_prompt() + f'User prompt: {user_prompt}' + f'\nAssistant response: '
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            streamer=self.streamer,
        )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return extract_json_from_text(decoded)
