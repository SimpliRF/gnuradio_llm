#
# This file is part of the GNU Radio LLM project.
#

from grc_dataset_logger.flowgraph_logger import FlowgraphLogger
from grc_dataset_logger.config import Config
from grc_dataset_logger.patches import patch_flowgraph


GRC_DATASET_LOGGER = FlowgraphLogger(Config())


def grc_hook():
    patch_flowgraph(GRC_DATASET_LOGGER)


grc_hook()
