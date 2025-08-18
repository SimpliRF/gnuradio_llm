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
    def __init__(self, model_name: str = 'TheBloke/Mistral-7B-Instruct-v0.1-GPTQ'):
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
            quantization_config=self.config,
            trust_remote_code=True
        )
        self.model.eval()
        self.streamer = TextStreamer(self.tokenizer)

    def generate(self, user_prompt: str, max_tokens: int = 1024) -> str:
        prompt = (
            get_system_prompt()
            f'\n\n### User prompt: {user_prompt}'
            f'\n\n### Assistant response: '
        )
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=3,
            early_stopping=True,
            streamer=self.streamer,
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
