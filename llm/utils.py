#
# This file is part of the GNU Radio LLM project.
#

import re
import json


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from a text string.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return json.dumps(data, indent=4)
        except json.JSONDecodeError:
            pass
    raise ValueError('Failed to extract valid JSON from text')
