#
# This file is part of the GNU Radio LLM project.
#

import os
import json
import base64

from typing import Any, Dict, Iterator

from datasets import Dataset


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
                    context = r.get('context', '')
                    if len(context):
                        context = decode_completion(r['context'])
                    history.append({
                        'prompt': r['prompt'],
                        'context': context,
                        'completion': decode_completion(r['completion']),
                    })

                i = 0
                while i + 1 <= len(history):
                    yield history[i]
                    i += 1


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
    return dataset # type: ignore
