#
# This file is part of the GNU Radio LLM project.
#

import pytest
import json

from pathlib import Path

from rich.console import Console

from flowgraph.schema import Flowgraph
from flowgraph.runner import FlowgraphRunner


def test_runner_from_valid_json():
    graph_path = Path('tests/mock_json/flowgraph_simple.json')
    graph = json.load(graph_path.open())
    flowgraph = Flowgraph(**graph)

    console = Console()

    runner = FlowgraphRunner(flowgraph, console)

    assert runner.tb is not None
    assert runner.generated_path is not None

    runner.tb.start()
    runner.tb.stop()
    runner.tb.wait()
