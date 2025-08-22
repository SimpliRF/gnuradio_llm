#
# This file is part of the GNU Radio LLM project.
#

import json
import threading
import datetime

from uuid import uuid4

from grc_dataset_logger.config import Config


class RuntimeLogger:
    """
    The runtime logger records traces from live flowgraphs deployed via GRC.
    """
    def __init__(self, config: Config = Config()):
        self.config = config
        self.lock = threading.Lock()
        self.config.trace_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = uuid4().hex[:8]

        trace_file_name = f'{self.session_id}.jsonl'
        self.traces_dir = self.config.trace_dir / 'actions'
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.traces_path = self.traces_dir / trace_file_name

        self.traces = []

    def _timestamp(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z'

    @staticmethod
    def _sanitize_for_json(data):
        if isinstance(data, (list, tuple, set)):
            return [RuntimeLogger._sanitize_for_json(item) for item in data]
        elif isinstance(data, dict):
            return {key: RuntimeLogger._sanitize_for_json(value) for key, value in data.items()}
        elif isinstance(data, (str, int, float, bool)):
            return data
        return str(data)

    def on_top_block_change(self, top_block, method, args, kwargs, result):
        with self.lock:
            if not method.startswith('set_') and not method.startswith('get_'):
                return

            self.traces.append({
                'flowgraph_id': str(top_block.__class__.__name__),
                'timestamp': self._timestamp(),
                'method': method,
                'args': self._sanitize_for_json(args),
                'kwargs': self._sanitize_for_json(kwargs),
                'result': self._sanitize_for_json(result),
            })

    def save_session(self):
        with self.lock:
            if not self.traces:
                print('No action traces saved this session')
                return

            with open(self.traces_path, 'w') as fp:
                for trace in self.traces:
                    json.dump(trace, fp)
                    fp.write('\n')

            print(f'---> Saved action traces to {self.traces_path}')
