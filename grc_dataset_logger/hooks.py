#
# This file is part of the GNU Radio LLM project.
#

from grc_dataset_logger.logger import GRCLogger
from grc_dataset_logger.config import Config
from grc_dataset_logger.patches import apply_patches


GRC_DATASET_LOGGER = GRCLogger(Config())


def grc_hook():
    apply_patches(GRC_DATASET_LOGGER)


grc_hook()
