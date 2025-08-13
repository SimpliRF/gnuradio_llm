#
# This file is part of the GNU Radio LLM project.
#

import pytest

from flowgraph.schema import Flowgraph


def test_flowgraph_validation():
    graph = {
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
        "connections": [{"from": "block1:output1", "to": "block2:input"}],
        "gui_config": {"enabled": True},
        "meta_info": {"description": "A test flowgraph", "tags": ["test"]}
    }

    flowgraph = Flowgraph(**graph)
    assert flowgraph.name == "Test Graph"
    assert len(flowgraph.blocks) == 2

    assert flowgraph.blocks[0].id == "block1"
    assert flowgraph.blocks[1].name == "Block 2"

    assert len(flowgraph.connections) == 1
    assert flowgraph.gui_config.enabled is True


def test_flowgraph_missing_parameters():
    graph = {
        "name": "Invalid Graph",
        "blocks": [
            {
                "id": "block1",
                "type": "source",
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

    with pytest.raises(Exception) as e:
        Flowgraph(**graph)

        assert "missing" in str(e.value), "Expected validation error for missing parameters"
