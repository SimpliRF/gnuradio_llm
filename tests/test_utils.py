#
# This file is part of the GNU Radio LLM project.
#

import pytest
import json

from llm.utils import extract_json_from_text


def test_extract_valid_json():
    text = 'Here is your result:\n\n{ "foo": 1, "bar": 2 }'
    result = extract_json_from_text(text)
    assert isinstance(result, str)
    assert '"foo": 1' in result
    assert '"bar": 2' in result


def test_extract_invalid_json():
    with pytest.raises(ValueError):
        text = 'Here is your result:\n\n{ "foo": 1, "bar": 2 '
        extract_json_from_text(text)
