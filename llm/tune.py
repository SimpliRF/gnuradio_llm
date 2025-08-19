#
# This file is part of the GNU Radio LLM project.
#

import os

import torch

from typing import Any

from peft import LoraConfig
from peft.mapping import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from llm.prompts import load_dataset, build_prompt


class ModelTrainer:
    def __init__(self,
                 model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 fallback_model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct',
                 dataset_dir: str = 'data',
                 output_dir: str = 'output',
                 hf_token_env: str = 'HUGGINGFACE_HUB_TOKEN'):
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.hf_token = os.environ.get(hf_token_env, None)

        self.included_schema = False
        self.model = self._load_model()

    def _apply_lora(self, model) -> Any:
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            task_type='CASUAL_LM',
            target_modules=[
                'q_proj',
                'v_proj',
                'k_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
        )
        return get_peft_model(model, self.peft_config)

    def _load_model(self) -> Any:
        if torch.cuda.is_available():
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type='nf4'
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map='auto',
                quantization_config=config
            )
            model.config.use_cache = False
            return self._apply_lora(model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.fallback_model_name,
            use_fast=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.fallback_model_name,
            device_map='cpu',
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.config.use_cache = False
        return self._apply_lora(model)

    @staticmethod
    def _make_formatting_func(tokenizer):
        def format_batch(batch):
            prompts = batch['prompt']
            completions = batch['completion']
            include_schema = batch.get('include_schema', [False] * len(prompts))

            texts = []

            zipped = zip(prompts, completions, include_schema)
            for prompt, completion, schema in zipped:
                text = build_prompt(
                    tokenizer,
                    prompt,
                    completion,
                    generation_prompt=False,
                    include_schema=bool(schema)
                )
                texts.append(text)
            return texts
        return format_batch

    def train(self, learning_rate: float = 2e-4, num_train_epochs: int = 50):
        dataset = load_dataset(self.dataset_dir)
        if torch.cuda.is_available():
            config = SFTConfig(
                output_dir=self.output_dir,
                max_seq_length=4096,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=True,
                lr_scheduler_type='cosine',
                warmup_ratio=0.0,
                weight_decay=0.0,
                logging_steps=10,
                save_steps=100,
            )
        else:
            config = SFTConfig(
                output_dir=self.output_dir,
                max_seq_length=4096,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=False,
                lr_scheduler_type='cosine',
                warmup_ratio=0.0,
                weight_decay=0.0,
                logging_steps=10,
                save_steps=100,
            )

        trainer = SFTTrainer(
            model=self.model,
            formatting_func=self._make_formatting_func(self.tokenizer),
            peft_config=self.peft_config,
            train_dataset=dataset,
            args=config,
        )
        trainer.train()
        trainer.save_model(self.output_dir)
