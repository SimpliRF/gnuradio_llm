#
# This file is part of the GNU Radio LLM project.
#

import os
import json
import torch
import base64

from typing import Any

from peft import LoraConfig
from peft.mapping import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


class ModelTrainer:
    def __init__(self,
                 model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 dataset_dir: str = 'data',
                 output_dir: str = 'qlora_data',
                 load_model: bool = True):
        self.model_name = model_name
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

        self.tokenizer = None
        self.model = None
        self.perft_config = None
        if load_model:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = self._load_model()
            self.model = self._apply_lora()

    def _load_model(self) -> Any:
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
            trust_remote_code=False
        )
        model.config.use_cache = False
        return model

    def _apply_lora(self) -> Any:
        if self.model is None:
            raise RuntimeError('Model must be loaded before applying LoRA.')
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            task_type='CASUAL_LM',
            target_modules=['all-linear'],
        )
        return get_peft_model(self.model, self.peft_config)

    def _load_dataset(self) -> Dataset:
        samples = []
        for file in os.listdir(self.dataset_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.dataset_dir, file), 'r') as fp:
                    data = json.load(fp)
                    completion = base64.b64decode(data['completion'])
                    completion = json.loads(completion)
                    samples.append({
                        'prompt': data['prompt'],
                        'completion': json.dumps(completion, indent=2)
                    })
        dataset = Dataset.from_list(samples)
        def format_example(example):
            return {'text': f'{example['prompt']}\n{example['completion']}'}

        return dataset.map(format_example)

    def train(self, learning_rate: float = 2e-4, num_train_epochs: int = 3):
        if self.model is None or self.peft_config is None:
            raise RuntimeError("Model and config must be loaded before training.")

        dataset = self._load_dataset()

        config = SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_steps=100,
        )

        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_config,
            train_dataset=dataset,
            args=config,
        )

        trainer.train()
        self.model.save_pretrained(os.path.join(self.output_dir, 'tuned_model'))
