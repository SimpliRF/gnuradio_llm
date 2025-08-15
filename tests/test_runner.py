#
# This file is part of the GNU Radio LLM project.
#

import pytest

from flowgraph.builder import FlowgraphBuilder
from flowgraph.runner import FlowgraphRunner


def test_runner_from_valid_json():
    json_data = '''
    '''

    graph = FlowgraphBuilder.from_json(json_data)
    runner = FlowgraphRunner(graph)
    runner._build()

    assert 'src' in runner.blocks
    assert 'throttle' in runner.blocks
    assert 'sink' in runner.blocks

    runner.tb.start()
    runner.tb.stop()
    runner.tb.wait()
