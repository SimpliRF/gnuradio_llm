#
# This file is part of the GNU Radio LLM project.
#

import re
import json


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from a text string.
    """
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            pass
    raise ValueError('Failed to extract valid JSON from text')
