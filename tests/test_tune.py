#
# This file is part of the GNU Radio LLM project.
#

import pytest

from llm.tune import ModelTrainer


def test_dataset_loading():
    dataset_dir = 'tests/mock_data'
    trainer = ModelTrainer(
        model_name='noop', dataset_dir=dataset_dir, load_model=False
    )
    dataset = trainer._load_dataset()

    assert len(dataset) > 0, "Dataset should not be empty"
    assert 'text' in dataset[0], "Dataset items should contain 'text' field"
    assert 'flowgraph' not in dataset[0], "Dataset items should not contain 'flowgraph' field"

    assert dataset[0]['prompt'] is not None, "Dataset items should contain a non-empty 'text' field"
    assert dataset[0]['response'] is not None, "Dataset items should contain a non-empty 'response' field"
