#
# This file is part of the GNU Radio LLM project.
#

import multiprocessing as mp

from multiprocessing import connection

from pathlib import Path
from rich.console import Console

from flowgraph.schema import Flowgraph, FlowgraphAction
from flowgraph.loader import generate_flowgraph
from flowgraph.remote import RemoteTopBlock


class FlowgraphController:
    def __init__(self, console: Console):
        self.console = console
        self.generated_path = None

        self.process = None
        self.parent_conn = None
        self.child_conn = None
        self.state = 'idle'

    def load_flowgraph(self, flowgraph: Flowgraph):
        self.generated_path = generate_flowgraph(flowgraph)
        self.state = 'loaded'
        self.console.print('🔧 Flowgraph loaded.')

    def _start_process(self):
        self.parent_conn, self.child_conn = mp.Pipe()
        self.process = mp.Process(
            target=RemoteTopBlock.entry_point,
            args=(self.generated_path, self.child_conn)
        )
        self.process.start()

        response = self.parent_conn.recv()
        if response.get('type') != 'status' or response.get('msg') != 'ready':
            raise RuntimeError(f'Failed to start remote process: {response}')

    def _send(self, msg: dict):
        if not self.parent_conn:
            raise RuntimeError('No connection to remote process.')
        self.parent_conn.send(msg)

        response_codes = ('started', 'stopped', 'set', 'get')
        response = self.parent_conn.recv()
        if response.get('type') == 'error':
            raise RuntimeError(response.get('err'))
        elif response.get('type') in response_codes:
            return response
        return response

    def start(self):
        if self.state == 'running':
            self.console.print('⚠️ Flowgraph is already running.')
            return

        if self.state != 'loaded':
            self.console.print('⚠️ Flowgraph is not loaded.')
            return

        if not self.generated_path:
            raise RuntimeError('No flowgraph loaded.')

        self._start_process()
        self._send({'type': 'start'})
        self.state = 'running'
        self.console.print('▶️ Flowgraph started.')

    def stop(self):
        if self.state != 'running':
            self.console.print('⚠️ Flowgraph is not running.')
            return

        if self.process is None:
            raise RuntimeError('No process to stop.')

        self._send({'type': 'stop'})
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()

        self.state = 'idle'
        self.console.print('⏹️ Flowgraph stopped.')

    def handle_action(self, action: FlowgraphAction):
        if action.action == 'start':
            self.start()
        elif action.action == 'stop':
            self.stop()
        elif action.action == 'block_set':
            self._send({
                'type': 'set',
                'method': action.method,
                'value': action.value
            })
            self.console.print(f'🔧 Set {action.method} to {action.value}')
        elif action.action == 'block_get':
            result = self._send({
                'type': 'get',
                'method': action.method
            })
            self.console.print(f'🔍 Get {action.method}: {result}')
        else:
            raise ValueError(f'Unknown action: {action.action}')
