#
# This file is part of the GNU Radio LLM project.
#

import os

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    output_dir: Path = Path(os.environ.get('GRC_DATASET_DIR', 'traces'))
