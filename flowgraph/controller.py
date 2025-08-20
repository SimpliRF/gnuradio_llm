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
            return 'No flowgraph loaded.'
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
        if not self.runner:
            return 'No flowgraph loaded'

        if action.action == 'start':
            self.start()
            return 'Flowgraph has started'

        elif action.action == 'stop':
            self.stop()
            return 'Flowgraph has stopped'

        elif action.action == 'block_set':
            if action.block_id:
                block = self.runner.get_block(action.block_id)
                if block is None:
                    return f'Block with ID {action.block_id} not found'

                if action.method is None:
                    return 'Missing method for set action'

                if action.value is None:
                    return 'Missing value for set action'

                method = getattr(block, action.method, None)
                if callable(method):
                    method(action.value)
                    return f'Successfully set: {action.method} = {action.value}'
                return f'Setter method {action.method} not found on block {action.block_id}'
            return f'Block {action.block_id} not found'

        elif action.action == 'block_get':
            if action.block_id:
                block = self.runner.get_block(action.block_id)
                if block is None:
                    return f'Block {action.block_id} not found'

                if action.method is None:
                    return 'Missing method for get action'

                method = getattr(block, action.method, None)
                if callable(method):
                    value = method()
                    return f'Successfully retrieved: {action.method} = {value}'
                return f'Getter method {action.method} not found on block {action.block_id}'
            return f'Block {action.block_id} not found'

        raise ValueError(f'Unsupported action: {action.action}')
