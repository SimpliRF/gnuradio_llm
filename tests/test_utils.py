#
# This file is part of the GNU Radio LLM project.
#

import pytest

from llm.utils import extract_json_from_text


def test_extract_valid_json():
    text = 'Here is your result:\n\n{ "foo": 1, "bar": 2 }'
    result = extract_json_from_text(text)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert 'foo' in result[0]
    assert 'bar' in result[0]

    text = ('Another one:\n\n{ "foo": 1, "bar": 2 }, '
            '{"derp": {"foo": 1}, "herp": 4}\n')
    result = extract_json_from_text(text)

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert 'foo' in result[0]
    assert 'bar' in result[0]
    assert 'derp' in result[1]
    assert 'herp' in result[1]


def test_extract_invalid_json():
    text = 'Here is your result:\n\n{ "foo": 1, "bar": 2 '
    result = extract_json_from_text(text)

    assert isinstance(result, list)
    assert len(result) == 0
