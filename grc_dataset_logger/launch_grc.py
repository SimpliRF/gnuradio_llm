#!/usr/bin/env python3
#
# This file is part of the GNU Radio LLM project.
#

import atexit

from grc_dataset_logger.flowgraph_logger import FlowgraphLogger
from grc_dataset_logger.config import Config
from grc_dataset_logger.patches import patch_flowgraph

from gnuradio.grc.main import main


GRC_DATASET_LOGGER = FlowgraphLogger(Config())


if __name__ == '__main__':
    patch_flowgraph(GRC_DATASET_LOGGER)
    atexit.register(GRC_DATASET_LOGGER.save_session)
    main()
