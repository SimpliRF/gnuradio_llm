#
# This file is part of the GNU Radio LLM project.
#

import pytest
import json

from flowgraph.schema import Flowgraph, FlowgraphAction
from llm.prompts import get_system_prompt, build_prompt, load_dataset


class DummyChatTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append((messages, tokenize, add_generation_prompt))

        def format_message(message):
            return f'{message["role"]}: {message["content"]}'

        rendered = '\n'.join(format_message(m) for m in messages)
        return rendered + f'\nADD_GEN:{str(add_generation_prompt)}'


class DummyTokenizer:
    pass


def test_system_prompt_contains_schema():
    prompt = get_system_prompt()

    flowgraph_schema = json.dumps(
        Flowgraph.model_json_schema(), separators=(',', ':')
    )
    action_schema = json.dumps(
        FlowgraphAction.model_json_schema(), separators=(',', ':')
    )

    assert flowgraph_schema.strip() in prompt
    assert action_schema.strip() in prompt


def test_build_prompt_with_tokenizer():
    tokenizer = DummyTokenizer()
    user_prompt = 'Create a flowgraph with a source and a sink'

    completion_json = '{ "ok": 1 }'

    output = build_prompt(
        tokenizer, user_prompt, completion_json, generation_prompt=False
    )

    assert '### Prompt: Create a flowgraph with a source and a sink' in output
    assert '### Completion:' in output
    assert '{ "ok": 1 }' in output

    output = build_prompt(
        tokenizer, user_prompt, completion_json
    )

    assert '### Completion:' in output
    assert '{ "ok": 1 }' not in output


def test_build_prompt_with_chat_tokenizer():
    tokenizer = DummyChatTokenizer()
    user_prompt = 'Create a flowgraph with a source and a sink'

    completion_json = '{ "ok": 1 }'

    output = build_prompt(
        tokenizer, user_prompt, completion_json, generation_prompt=False
    )

    assert 'user: Create a flowgraph with a source and a sink' in output
    assert 'assistant: { "ok": 1 }' in output
    assert 'ADD_GEN:False' in output

    output = build_prompt(
        tokenizer, user_prompt, completion_json
    )

    assert 'user: Create a flowgraph with a source and a sink' in output
    assert 'ADD_GEN:True' in output
