#
# This file is part of the GNU Radio LLM project.
#

import json

from typing import Any, List, Dict


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract JSON content from a text string.
    """
    results = []

    depth = 0
    start_idx = None

    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            else:
                if ch == '\\':
                    escaped = True
                elif ch == '"':
                    in_string = False
            continue

        if ch == '"':
            in_string = True
            escaped = False
            continue

        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidate = text[start_idx:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
    return results
