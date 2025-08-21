#
# This file is part of the GNU Radio LLM project.
#

import json
import threading
import datetime

from uuid import uuid4

from grc_dataset_logger.config import Config


class FlowgraphLogger:
    """
    The flowgraph logger records changes to GRC flowgraphs before execution.
    """
    def __init__(self, config: Config = Config()):
        self.config = config
        self.lock = threading.Lock()
        self.config.trace_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = uuid4().hex[:8]

        trace_file_name = f'{self.session_id}.jsonl'
        self.traces_dir = self.config.trace_dir / 'flowgraph'
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.traces_path = self.traces_dir / trace_file_name

        self.traces = []
        self.prev_snapshot = None

    def _timestamp(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z'

    def on_flowgraph_change(self, flowgraph, method, args, kwargs):
        with self.lock:
            snapshot_1 = flowgraph.export_data()
            if snapshot_1 == self.prev_snapshot:
                return

            snapshot_0 = self.prev_snapshot
            self.prev_snapshot = snapshot_1
            self.traces.append({
                'flowgraph_id': flowgraph.get_option('id'),
                'timestamp': self._timestamp(),
                'snapshot_0': snapshot_0,
                'snapshot_1': snapshot_1,
            })

    def save_session(self):
        with self.lock:
            if not self.traces:
                print('No flowgraph traces saved this session')
                return

            with open(self.traces_path, 'w') as fp:
                for trace in self.traces:
                    json.dump(trace, fp)
                    fp.write('\n')

            print(f'---> Saved flowgraph traces to {self.traces_path}')
