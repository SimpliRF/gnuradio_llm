#
# This file is part of the GNU Radio LLM project.
#

import pytest
import os
import json

from pathlib import Path

from rich.console import Console

from flowgraph.schema import Flowgraph, FlowgraphAction
from flowgraph.controller import FlowgraphController


@pytest.mark.skipif(
    not os.environ.get('DISPLAY'),
    reason='Requires a display (Qt GUI) to run'
)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_flowgraph_controller():
    graph_path = Path('tests/mock_json/flowgraph_simple.json')
    graph = json.load(graph_path.open())
    flowgraph = Flowgraph(**graph)

    console = Console()
    controller = FlowgraphController(console)

    assert 'idle' in controller.state

    controller.load_flowgraph(flowgraph)

    assert 'loaded' in controller.state

    controller.start()
    assert 'running' in controller.state

    controller.stop()
    assert 'idle' in controller.state


@pytest.mark.skipif(
    not os.environ.get('DISPLAY'),
    reason='Requires a display (Qt GUI) to run'
)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_flowgraph_controller_action():
    graph_path = Path('tests/mock_json/flowgraph_callbacks.json')
    graph = json.load(graph_path.open())
    flowgraph = Flowgraph(**graph)

    console = Console(record=True)
    controller = FlowgraphController(console)

    controller.load_flowgraph(flowgraph)

    start_action_json = '''
    {
        "action": "start"
    }
    '''

    start_action = FlowgraphAction.model_validate_json(start_action_json)
    controller.handle_action(start_action)

    assert 'running' in controller.state

    set_action_json = '''
    {
        "action": "block_set",
        "method": "set_samp_rate",
        "value": 33000.0
    }
    '''

    set_action = FlowgraphAction.model_validate_json(set_action_json)
    controller.handle_action(set_action)

    assert 'running' in controller.state
    assert 'Set set_samp_rate to 33000.0' in console.export_text()

    get_action_json = '''
    {
        "action": "block_get",
        "method": "get_samp_rate"
    }
    '''

    get_action = FlowgraphAction.model_validate_json(get_action_json)
    controller.handle_action(get_action)

    assert 'running' in controller.state
    assert 'Get get_samp_rate: ' in console.export_text()

    stop_action_json = '''
    {
        "action": "stop"
    }
    '''

    stop_action = FlowgraphAction.model_validate_json(stop_action_json)
    controller.handle_action(stop_action)

    assert 'idle' in controller.state
