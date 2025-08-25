#
# This file is part of the GNU Radio LLM project.
#

import pytest
import json

from pathlib import Path

from flowgraph.schema import Flowgraph
from llm.prompts import get_system_prompt


def test_system_prompt_contains_schema():
    prompt = get_system_prompt(include_schema=True)
    assert "Flowgraph Schema" in prompt
    assert "Action Schema" in prompt


def test_flowgraph_validation():
    graph_path = Path('tests/mock_data/flowgraph_simple.json')
    graph = json.load(graph_path.open())
    flowgraph = Flowgraph(**graph)

    assert len(flowgraph.options) == 2
    assert len(flowgraph.blocks) == 3
    assert len(flowgraph.connections) == 2

    assert 'test' in flowgraph.options['parameters']['id']


def test_flowgraph_invalid():
    graph = {
        "name": "Invalid Graph",
        "meta_info": {"description": "Invalid format"}
    }

    with pytest.raises(Exception) as e:
        Flowgraph(**graph)

        assert 'missing' in str(e.value)
