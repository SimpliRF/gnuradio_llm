#
# This file is part of the GNU Radio LLM project.
#

import pytest

from rich.console import Console

from flowgraph.schema import Flowgraph, FlowgraphAction
from flowgraph.controller import FlowgraphController


TEST_FLOWGRAPH_JSON = '''
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
            }
        },
        {
            "id": "throttle",
            "name": "Throttle",
            "type": "throttle",
            "parameters": {
                "itemsize": 4,
                "samples_per_sec": 32000.0
            }
        },
        {
            "id": "sink",
            "name": "Null Sink",
            "type": "null_sink",
            "parameters": {
                "sizeof_stream_item": 4
            }
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


def test_flowgraph_controller():
    console = Console()
    controller = FlowgraphController(console)

    assert controller.console == console
    assert controller.state == 'idle'
    assert controller.flowgraph is None
    assert controller.runner is None

    graph = Flowgraph.model_validate_json(TEST_FLOWGRAPH_JSON)
    controller.load_flowgraph(graph)

    assert controller.flowgraph == graph
    assert controller.state == 'loaded'
    assert controller.runner is not None
    assert controller.runner.flowgraph == graph

    controller.start()
    assert controller.state == 'running'

    controller.stop()
    assert controller.state == 'stopped'


def test_flowgraph_controller_action():
    console = Console()
    controller = FlowgraphController(console)

    graph = Flowgraph.model_validate_json(TEST_FLOWGRAPH_JSON)
    controller.load_flowgraph(graph)

    start_action_json = '''
    {
        "action": "start"
    }
    '''

    start_action = FlowgraphAction.model_validate_json(start_action_json)
    response = controller.handle_action(start_action)

    assert 'Flowgraph has started' in response

    stop_action_json = '''
    {
        "action": "stop"
    }
    '''

    stop_action = FlowgraphAction.model_validate_json(stop_action_json)
    response = controller.handle_action(stop_action)

    assert 'Flowgraph has stopped' in response

    set_action_json = '''
    {
        "action": "block_set",
        "block_id": "throttle",
        "method": "set_sample_rate",
        "value": 33000.0
    }
    '''

    set_action = FlowgraphAction.model_validate_json(set_action_json)
    response = controller.handle_action(set_action)

    assert 'Success' in response
    assert 'set_sample_rate' in response
    assert '33000.0' in response

    get_action_json = '''
    {
        "action": "block_get",
        "block_id": "throttle",
        "method": "get_sample_rate"
    }
    '''

    get_action = FlowgraphAction.model_validate_json(get_action_json)
    response = controller.handle_action(get_action)

    assert 'not found on block' in response
