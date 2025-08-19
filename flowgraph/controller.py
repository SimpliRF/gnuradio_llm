#
# This file is part of the GNU Radio LLM project.
#

import time

from typing import Optional
from rich.console import Console
from flowgraph.runner import FlowgraphRunner
from flowgraph.schema import Flowgraph, FlowgraphAction


class FlowgraphController:
    def __init__(self, console: Console):
        self.console = console
        self.runner: Optional[FlowgraphRunner] = None
        self.flowgraph: Optional[Flowgraph] = None
        self.state: str = 'idle'
        self.start_time: Optional[float] = None

    def load_flowgraph(self, flowgraph: Flowgraph):
        """
        Load and validate a flowgraph from JSON or LLM output.
        """
        self.flowgraph = flowgraph
        self.runner = FlowgraphRunner(flowgraph, self.console)
        self.state = 'loaded'

    def start(self):
        if not self.runner:
            raise RuntimeError("FlowgraphRunner is not loaded.")
        if self.state == 'running':
            self.console.print("Flowgraph is already running.")

        self.runner.start()
        self.state = 'running'
        self.start_time = time.time()
        self.console.print("Flowgraph started.")

    def stop(self):
        if self.runner and self.state == 'running':
            self.runner.tb.stop()
            self.state = 'stopped'
            self.console.print("Flowgraph stopped.")

    def status(self) -> str:
        if not self.runner:
            return "No flowgraph loaded."
        duration = time.time() - self.start_time if self.start_time else 0
        return f'State: {self.state} | Uptime: {duration:.2f} seconds'

    def reset(self):
        if self.runner and self.state == 'running':
            self.stop()

        self.runner = None
        self.flowgraph = None
        self.state = 'idle'
        self.start_time = None
        self.console.print('Flowgraph reset')

    def handle_action(self, action: FlowgraphAction) -> str:
        if action.action == 'start':
            self.start()
            return 'flowgraph has started'
        elif action.action == 'stop':
            self.stop()
            return 'flowgraph has stopped'
        elif action.action == 'set':
            if action.block_id and action.parameter:
                # TODO: Set the parameter of a block
                return 'flowgraph parameter has been set'
            if action.block_id is None:
                return 'missing block ID for set action'
            if action.parameter is None:
                return 'missing parameter for set action'
        elif action.action == 'get':
            if action.block_id and action.parameter:
                # TODO: Get the parameter of a block
                value = ''
                return (f'flowgraph parameter has been retrieved: ' +
                        f'{action.parameter} = {value}')
            if action.block_id is None:
                return 'missing block ID for get action'
            if action.parameter is None:
                return 'missing parameter for get action'

        raise ValueError(f'Unsupported action: {action.action}')
