#!/usr/bin/env python3
#
# This file is part of the GNU Radio LLM project.
#

import sys

from pathlib import Path

from grc_dataset_logger.action_logger import ActionLogger
from grc_dataset_logger.config import Config
from grc_dataset_logger.patches import load_top_block, patch_top_block


ACTION_LOGGER = ActionLogger(Config())


def execute_script(script_path: Path):
    main_func, top_block_cls = load_top_block(script_path)

    patch_top_block(top_block_cls, ACTION_LOGGER)

    result = main_func(top_block_cls)

    ACTION_LOGGER.save_session()
    return result


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print('Usage: python launch_top_block.py <path_to_top_block>')
        sys.exit(1)

    result = execute_script(Path(path))
    sys.exit(result)
