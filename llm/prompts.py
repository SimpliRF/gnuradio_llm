#
# This file is part of the GNU Radio LLM project.
#

import os
import json
import base64

from datasets import Dataset


SYSTEM_PROMPT_PREFIX = '''
You are an assistant that generates GNU Radio flowgraphs.
You must respond only with JSON matching one of the following schemas:
- Flowgraph: when creating or modifying a flowgraph.
- FlowgraphAction: when performing control actions like start, stop, and more.

Do not include any explanations or extra text. Return exactly ONE JSON object.
'''


def get_system_prompt() -> str:
    return f'{SYSTEM_PROMPT_PREFIX}\n\n'


def build_prompt(tokenizer,
                 user_prompt: str,
                 completion_json: str = '',
                 generation_prompt: bool = True) -> str:
    """
    Build a consistent prompt for training or inference.
    """
    completion_json = completion_json.strip()
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {'role': 'system', 'content': get_system_prompt()},
            {'role': 'user', 'content': user_prompt},
        ]

        if not generation_prompt:
            messages.append({'role': 'assistant', 'content': completion_json})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt
        )
        return prompt

    system_prompt = get_system_prompt()
    if generation_prompt:
        return f'{system_prompt}\n\n### Prompt: {user_prompt}\n\n### Completion: '
    else:
        return f'{system_prompt}\n\n### Prompt: {user_prompt}\n\n### Completion: {completion_json}'


def load_dataset(tokenizer, dataset_dir: str) -> Dataset:
    """
    Load the dataset from the specified directory.
    """
    samples = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            with open(os.path.join(dataset_dir, filename), 'r') as fp:
                data = json.load(fp)
                completion = base64.b64decode(data['completion'])
                completion = json.loads(completion)
                completion_str = json.dumps(completion, separators=(',', ':'))
                samples.append({
                    'prompt': data['prompt'],
                    'completion': completion_str
                })
    dataset = Dataset.from_list(samples)
    def map_to_text(text):
        return {
            'text': build_prompt(
                tokenizer,
                text['prompt'],
                text['completion'],
                generation_prompt=False
            )
        }
    dataset = dataset.map(map_to_text, remove_columns=['prompt', 'completion'])
    return dataset
