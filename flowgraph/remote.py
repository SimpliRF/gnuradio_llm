#
# This file is part of the GNU Radio LLM project.
#

from multiprocessing import connection

from typing import Dict, Any
from pathlib import Path

from PyQt5 import Qt, QtCore # type: ignore

from flowgraph.loader import load_top_block


class RemoteTopBlock:
    def __init__(self, generated_path: Path, connection: connection.Connection):
        self.generated_path = generated_path
        self.connection = connection
        self.app = Qt.QApplication([])
        self.timer = QtCore.QTimer()

        tb_cls = self._load_tb_cls()
        self.tb = tb_cls()

    @staticmethod
    def entry_point(generated_path: Path, connection: connection.Connection):
        remote_top_block = RemoteTopBlock(generated_path, connection)
        remote_top_block.main()

    def _poll_timer(self):
        self.timer.setInterval(30)
        def poll():
            try:
                if self.connection and self.connection.poll():
                    command = self.connection.recv()
                    self._handle_command(command)
            except (EOFError, BrokenPipeError):
                self.connection.close()
                self.app.quit()
        self.timer.timeout.connect(poll)
        self.timer.start()

    def _load_tb_cls(self):
        _, tb_cls = load_top_block(self.generated_path)
        return tb_cls

    def _send(self, msg: Dict[str, str]):
        self.connection.send(msg)

    def _handle_command(self, cmd: Dict[str, Any]):
        command_type = cmd.get('type')
        try:
            if command_type == 'start':
                self.tb.start()
                self._send({'type': 'started'})
            elif command_type == 'stop':
                self.tb.stop()
                self.tb.wait()
                self._send({'type': 'stopped'})
            elif command_type == 'set':
                method = getattr(self.tb, cmd['method'], None)
                if callable(method):
                    method(cmd['value'])
                    self._send({
                        'type': 'set',
                        'method': cmd['method'],
                        'value': cmd['value']
                    })
                else:
                    self._send({
                        'type': 'error',
                        'err': f'Unknown method: {cmd["method"]}'
                    })
            elif command_type == 'get':
                method = getattr(self.tb, cmd['method'], None)
                if callable(method):
                    result = method()
                    self._send({
                        'type': 'get',
                        'method': cmd['method'],
                        'result': str(result)
                    })
                else:
                    self._send({
                        'type': 'error',
                        'err': f'Unknown method: {cmd["method"]}'
                    })
            else:
                self._send({
                    'type': 'error',
                    'err': f'Unknown command: {command_type}'
                })
        except Exception as e:
            self._send({
                'type': 'error',
                'err': f'Failed to handle command: {e}'
            })

    def main(self):
        self._poll_timer()
        self._send({'type': 'status', 'msg': 'ready'})
        try:
            self.app.exec()
        finally:
            self.tb.stop()
            self.tb.wait()
