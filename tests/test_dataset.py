#
# This file is part of the GNU Radio LLM project.
#

import pytest

from typing import Iterator

from llm.dataset import load_dataset_jsonl, load_dataset


def test_load_dataset_jsonl():
    dataset = load_dataset_jsonl('tests/mock_datasets')

    assert isinstance(dataset, Iterator)

    first_item = next(dataset)
    assert 'history' in first_item
    assert isinstance(first_item['history'], list)


def test_load_dataset():
    dataset = load_dataset('tests/mock_datasets')

    assert len(dataset) == 1
    assert 'history' in dataset[0]

    chain = dataset[0]['history']
    pair = chain[0]
    assert 'prompt' in pair
    assert 'completion' in pair

    assert 'generate a null sink' in pair['prompt']
    assert '"blocks":[' in pair['completion']
    assert '"connections":[' in pair['completion']
