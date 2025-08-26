#
# This file is part of the GNU Radio LLM project.
#

import pytest
import json

from pathlib import Path

from rich.console import Console

from flowgraph.schema import Flowgraph, FlowgraphAction
from flowgraph.controller import FlowgraphController


def test_flowgraph_controller():
    graph_path = Path('tests/mock_json/flowgraph_simple.json')
    graph = json.load(graph_path.open())
    flowgraph = Flowgraph(**graph)

    console = Console()
    controller = FlowgraphController(console)

    assert controller.console == console
    assert controller.state == 'idle'
    assert controller.flowgraph is None
    assert controller.runner is None

    controller.load_flowgraph(flowgraph)

    assert controller.flowgraph == flowgraph
    assert controller.state == 'loaded'
    assert controller.runner is not None
    assert controller.runner.flowgraph == flowgraph

    controller.start()
    assert controller.state == 'running'

    controller.stop()
    assert controller.state == 'stopped'


# def test_flowgraph_controller_action():
#     graph_path = Path('tests/mock_data/flowgraph_simple.json')
#     graph = json.load(graph_path.open())
#     flowgraph = Flowgraph(**graph)
# 
#     console = Console()
#     controller = FlowgraphController(console)
# 
#     controller.load_flowgraph(flowgraph)
# 
#     start_action_json = '''
#     {
#         "action": "start"
#     }
#     '''
# 
#     start_action = FlowgraphAction.model_validate_json(start_action_json)
#     response = controller.handle_action(start_action)
# 
#     assert 'Flowgraph has started' in response
# 
#     stop_action_json = '''
#     {
#         "action": "stop"
#     }
#     '''
# 
#     stop_action = FlowgraphAction.model_validate_json(stop_action_json)
#     response = controller.handle_action(stop_action)
# 
#     assert 'Flowgraph has stopped' in response
# 
#     set_action_json = '''
#     {
#         "action": "block_set",
#         "method": "set_sample_rate",
#         "value": 33000.0
#     }
#     '''
# 
#     set_action = FlowgraphAction.model_validate_json(set_action_json)
#     response = controller.handle_action(set_action)
# 
#     assert 'Success' in response
#     assert 'set_sample_rate' in response
#     assert '33000.0' in response
# 
#     get_action_json = '''
#     {
#         "action": "block_get",
#         "method": "get_sample_rate"
#     }
#     '''
# 
#     get_action = FlowgraphAction.model_validate_json(get_action_json)
#     response = controller.handle_action(get_action)
