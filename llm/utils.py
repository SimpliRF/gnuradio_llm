#
# This file is part of the GNU Radio LLM project.
#

import re
import json

from typing import Any, List


def extract_json_from_text(text: str) -> List[Any]:
    """
    Extract JSON content from a text string.
    """
    result = []
    record = set()

    matches = re.findall(
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL
    )
    for m in matches:
        try:
            obj = json.loads(m)
            key = json.dumps(obj, sort_keys=True)
            if key not in record:
                record.add(key)
                result.append(obj)
        except json.JSONDecodeError:
            pass

    inline_matches = re.findall(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
    for m in inline_matches:
        try:
            obj = json.loads(m)
            key = json.dumps(obj, sort_keys=True)
            if key not in record:
                record.add(key)
                result.append(obj)
        except json.JSONDecodeError:
            pass
    return result
