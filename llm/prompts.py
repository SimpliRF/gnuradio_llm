#
# This file is part of the GNU Radio LLM project.
#

import os
import json
import base64

from typing import Any, Dict, Iterator

from datasets import Dataset

from flowgraph.schema import Flowgraph, FlowgraphAction


SYSTEM_PROMPT_PREFIX = '''
You are an assistant that generates and controls GNU Radio flowgraphs.
You must respond only with JSON matching one of the following schemas:
- Flowgraph: when creating or modifying a flowgraph.
- FlowgraphAction: when performing control actions like start, stop, and more.
Do not include any explanations or extra text. Return exactly ONE JSON object.
'''


def get_system_prompt(include_schema: bool = False) -> str:
    system_prompt = f'{SYSTEM_PROMPT_PREFIX}\n\n'
    if not include_schema:
        return system_prompt

    flowgraph_schema = json.dumps(
        Flowgraph.model_json_schema(), separators=(',', ':')
    )
    action_schema = json.dumps(
        FlowgraphAction.model_json_schema(), separators=(',', ':')
    )
    return (f'{system_prompt}' +
            f'### Flowgraph Schema:\n{flowgraph_schema}\n\n' +
            f'### Action Schema:\n{action_schema}\n')


def build_prompt(tokenizer,
                 user_prompt: str,
                 completion_json: str = '',
                 include_schema: bool = False,
                 generation_prompt: bool = True) -> str:
    """
    Build a consistent prompt for inference.
    """
    completion_json = completion_json.strip()
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {'role': 'system', 'content': get_system_prompt(include_schema)},
            {'role': 'user', 'content': user_prompt},
        ]

        if not generation_prompt:
            messages.append({'role': 'assistant', 'content': completion_json})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt
        )
        return prompt

    system_prompt = get_system_prompt(include_schema)
    if generation_prompt:
        return (f'{system_prompt}\n\n### Prompt: '
                f'{user_prompt}\n\n### Completion: ')
    else:
        return (f'{system_prompt}\n\n### Prompt: '
                f'{user_prompt}\n\n### Completion: {completion_json}')


def build_chained_prompt(tokenizer,
                         history: list[tuple[str, str]],
                         include_schema: bool = False,
                         generation_prompt: bool = True) -> str:
    """
    Build a chained prompt from a sequence of (user_prompt, completion_json)
    pairs.
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {'role': 'system', 'content': get_system_prompt(include_schema)}
        ]

        for user_prompt, completion_json in history[:-1]:
            completion_json = completion_json.strip()
            messages.append({'role': 'user', 'content': user_prompt})
            messages.append({'role': 'assistant', 'content': completion_json})

        user_prompt = history[-1][0]
        messages.append({'role': 'user', 'content': user_prompt})

        if not generation_prompt:
            completion_json = history[-1][1].strip()
            messages.append({'role': 'assistant', 'content': completion_json})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt
        )
        return prompt

    system_prompt = get_system_prompt(include_schema)
    result = system_prompt + '\n\n'
    for user_prompt, completion_json in history[:-1]:
        result += f'### Prompt: {user_prompt}\n\n'
        result += f'### Completion: {completion_json.strip()}\n\n'

    user_prompt = history[-1][0]
    result += f'### Prompt: {user_prompt}\n\n'

    if not generation_prompt:
        completion_json = history[-1][1].strip()
        result += f'### Completion: {completion_json}'

    return result


def decode_completion(completion_json: str) -> str:
    """
    Decode a base64-encoded JSON completion string.
    """
    completion_json = completion_json.strip()
    completion_json = base64.b64decode(completion_json).decode('utf-8')
    completion = json.loads(completion_json)
    return json.dumps(completion, separators=(',', ':'))


def load_dataset_jsonl(dataset_dir: str) -> Iterator[Dict[str, Any]]:
    for filename in os.listdir(dataset_dir):
        if not filename.endswith('.jsonl'):
            continue
        path = os.path.join(dataset_dir, filename)

        with open(path, 'r') as fp:
            for line in fp:
                line = line.strip()
                data = json.loads(line)

                if not isinstance(data, list):
                    continue

                history = []
                for r in data:
                    history.append({
                        'prompt': r['prompt'],
                        'completion': decode_completion(r['completion']),
                    })
                if history:
                    yield {'history': history}


def load_dataset(dataset_dir: str, cache_dir: str = 'dataset_cache') -> Dataset:
    """
    Load the dataset from the specified directory.
    """
    dataset = Dataset.from_generator(
        load_dataset_jsonl,
        gen_kwargs={'dataset_dir': dataset_dir},
        cache_dir=cache_dir,
        keep_in_memory=False
    )

    def add_schema_flag(example, index):
        example['include_schema'] = (index == 0)
        return example

    return dataset.map( # type: ignore
        add_schema_flag,
        with_indices=True,
        batch_size=8,
        keep_in_memory=False
    )
