#
# This file is part of the GNU Radio LLM project.
#

import pytest

from rich.console import Console

from flowgraph.schema import Flowgraph
from flowgraph.runner import FlowgraphRunner


def test_runner_from_valid_json():
    json_data = '''
    {
        "name": "Test Graph",
        "blocks": [
            {
                "id": "src",
                "name": "Signal Source",
                "type": "sig_source_f",
                "parameters": {
                    "sampling_freq": 32000.0,
                    "wave_freq": 1000.0,
                    "ampl": 1.0,
                    "waveform": 0
                },
                "inputs": [],
                "outputs": ["out"]
            },
            {
                "id": "throttle",
                "name": "Throttle",
                "type": "throttle",
                "parameters": {
                    "itemsize": 4,
                    "samples_per_sec": 32000.0
                },
                "inputs": ["in"],
                "outputs": ["out"]
            },
            {
                "id": "sink",
                "name": "Null Sink",
                "type": "null_sink",
                "parameters": {
                    "sizeof_stream_item": 4
                },
                "inputs": ["in"],
                "outputs": []
            }
        ],
        "connections": [
            {"from": "src:out", "to": "throttle:in"},
            {"from": "throttle:out", "to": "sink:in"}
        ],
        "gui_config": {"enabled": false},
        "meta_info": {"description": "A test flowgraph", "tags": ["test"]}
    }
    '''

    console = Console()

    graph = Flowgraph.model_validate_json(json_data)
    runner = FlowgraphRunner(graph, console)
    runner._build()

    assert 'src' in runner.blocks
    assert 'throttle' in runner.blocks
    assert 'sink' in runner.blocks

    runner.tb.start()
    runner.tb.stop()
    runner.tb.wait()
