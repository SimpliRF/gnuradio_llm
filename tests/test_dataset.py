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
    assert isinstance(first_item, dict)
    assert 'prompt' in first_item
    assert 'completion' in first_item
    assert 'context' in first_item


def test_load_dataset():
    dataset = load_dataset('tests/mock_datasets')

    assert len(dataset) == 1
    assert 'prompt' in dataset[0]
    assert 'completion' in dataset[0]
    assert 'context' in dataset[0]

    assert 'generate a null sink' in dataset[0]['prompt']
    assert '"blocks":[' in dataset[0]['completion']
    assert '"connections":[' in dataset[0]['completion']
