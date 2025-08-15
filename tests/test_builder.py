#
# This file is part of the GNU Radio LLM project.
#

import pytest

from flowgraph.builder import FlowgraphBuilder


def test_builder_from_valid_json():
    json_data = '''
    {
        "name": "Test Graph",
        "blocks": [
            {
                "id": "block1",
                "name": "Block 1",
                "type": "source",
                "parameters": {"param1": "value1"},
                "inputs": [],
                "outputs": ["output1"]
            },
            {
                "id": "block2",
                "name": "Block 2",
                "type": "sink",
                "parameters": {},
                "inputs": ["output1"],
                "outputs": []
            }
        ],
        "connections": [
            {"from": "block1:output1", "to": "block2:input"}
        ],
        "gui_config": {"enabled": true},
        "meta_info": {"description": "A test flowgraph", "tags": ["test"]}
    }
    '''
    flowgraph = FlowgraphBuilder.from_json(json_data)
    assert flowgraph.name == "Test Graph"
    assert len(flowgraph.blocks) == 2

    assert flowgraph.blocks[0].id == "block1"
    assert flowgraph.blocks[1].name == "Block 2"

    assert len(flowgraph.connections) == 1
    assert flowgraph.gui_config.enabled is True


def test_builder_from_invalid_json():
    json_data = '''
    {
        "name": "Invalid Graph",
        "blocks": [
            {
                "id": "block1",
                "type": "source"
            },
            {
                "id": "block2",
                "name": "Block 2",
                "type": "sink",
                "inputs": ["output1"],
                "outputs": []
            }
        ],
        "connections": [],
        "gui_config": {"enabled": True},
        "meta_info": {"description": "A test flowgraph with missing parameters"}
    }
    '''

    with pytest.raises(Exception) as e:
        FlowgraphBuilder.from_json(json_data)

        assert "missing" in str(e.value), "Expected validation error for missing parameters"
