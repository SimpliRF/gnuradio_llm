#
# This file is part of the GNU Radio LLM project.
#

import os

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = PROJECT_ROOT / 'traces'


@dataclass
class Config:
    trace_dir: Path = Path(os.environ.get('TRACE_DIR', DEFAULT_TRACE_DIR))
