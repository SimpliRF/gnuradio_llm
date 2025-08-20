#
# This file is part of the GNU Radio LLM project.
#

import os
import json
import base64
import threading

from typing import Dict, Any

from grc_dataset_logger.config import Config


class GRCLogger:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.lock = threading.Lock()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_session(self):
        pass

    def _get_completion(self, flowgraph) -> str:
        pass

    def _classify_flowgraph_action(self, method, args, kwargs) -> Dict[str, Any]:
        pass

    def _classify_block_action(self, method, args, kwargs) -> Dict[str, Any]:
        pass

    def on_flowgraph_change(self, flowgraph, method, args, kwargs):
        with self.lock:
            pass

    def on_block_param_change(self, block, method, args, kwargs):
        with self.lock:
            pass
