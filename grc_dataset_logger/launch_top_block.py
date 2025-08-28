#!/usr/bin/env python3
#
# This file is part of the GNU Radio LLM project.
#

import json
import base64

from pathlib import Path

from flowgraph.loader import load_top_block

from grc_dataset_logger.runtime_logger import RuntimeLogger
from grc_dataset_logger.config import Config
from grc_dataset_logger.patches import patch_top_block


ACTION_LOGGER = RuntimeLogger(Config())


def execute_script(script_path: Path, flowgraph_json: str):
    flowgraph_json = base64.b64decode(flowgraph_json).decode()
    flowgraph_data = json.loads(flowgraph_json)

    ACTION_LOGGER.load_flowgraph(flowgraph_data)

    main_func, top_block_cls = load_top_block(script_path)

    patch_top_block(top_block_cls, ACTION_LOGGER)

    result = main_func(top_block_cls)

    ACTION_LOGGER.save_session()
    return result
