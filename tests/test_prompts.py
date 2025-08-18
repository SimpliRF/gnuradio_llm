#
# This file is part of the GNU Radio LLM project.
#

import pytest
import json

from llm.prompts import get_system_prompt
from flowgraph.schema import Flowgraph


def test_system_prompt_contains_schema():
    prompt = get_system_prompt()
    schema = json.dumps(Flowgraph.model_json_schema(), indent=2)
    assert schema.strip() in prompt
