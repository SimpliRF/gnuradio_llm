#
# This file is part of the GNU Radio LLM project.
#

import os

import torch

from typing import Any

from peft import LoraConfig, prepare_model_for_kbit_training # type: ignore
from peft.mapping import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from llm.prompts import build_prompt
from llm.dataset import load_dataset


class ModelTrainer:
    def __init__(self,
                 model_name: str = 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
                 dataset_dir: str = 'dataset',
                 output_dir: str = 'output',
                 hf_token_env: str = 'HUGGINGFACE_HUB_TOKEN'):
        self.model_name = model_name
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.hf_token = os.environ.get(hf_token_env, None)

        self.model = self._load_model()

    def _apply_lora(self, model) -> Any:
        self.peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=[
                'q_proj',
                'v_proj',
                'k_proj',
                'o_proj',
            ],
            use_rslora=True,
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
                quantization_config=config,
                low_cpu_mem_usage=True
            )
            model.config.use_cache = False
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
            return self._apply_lora(model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
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
            contexts = batch['context']
            completions = batch['completion']

            result = []
            for p, ctx, comp in zip(prompts, contexts, completions):
                prompt = build_prompt(tokenizer, p, ctx, comp)
                result.append(prompt)
            return result

        return format_batch

    def train(self,
              max_seq_length: int = 2048,
              learning_rate: float = 2e-4,
              num_train_epochs: int = 5):

        dataset = load_dataset(self.dataset_dir)

        if torch.cuda.is_available():
            config = SFTConfig(
                output_dir=self.output_dir,
                max_seq_length=max_seq_length,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                save_steps=100,
            )
        else:
            config = SFTConfig(
                output_dir=self.output_dir,
                max_seq_length=max_seq_length,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                fp16=False,
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
