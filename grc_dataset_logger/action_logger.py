#
# This file is part of the GNU Radio LLM project.
#

import threading

from grc_dataset_logger.config import Config


class ActionLogger:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.lock = threading.Lock()

    def on_top_block_change(self, top_block, method, args, kwargs):
        with self.lock:
            pass
