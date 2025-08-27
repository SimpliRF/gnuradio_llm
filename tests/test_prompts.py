#
# This file is part of the GNU Radio LLM project.
#

import pytest

from llm.prompts import build_prompt


class DummyChatTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append((messages, tokenize, add_generation_prompt))

        def format_message(message):
            return f'{message['role']}: {message['content']}'

        rendered = '\n'.join(format_message(m) for m in messages)
        return rendered + f'\nADD_GEN:{str(add_generation_prompt)}'


def test_build_prompt_with_chat_tokenizer():
    tokenizer = DummyChatTokenizer()
    user_prompt = 'Create a flowgraph with a source and a sink'

    context_json = None
    completion_json = None

    output = build_prompt(
        tokenizer, user_prompt, context_json, completion_json
    )

    assert 'user: Create a flowgraph with a source and a sink' in output
    assert 'ADD_GEN:True' in output

    context_json = '{ "flowgraph": { "nodes": [], "edges": [] } }'
    completion_json = 'assistant: { "ok": 1 }'

    output = build_prompt(
        tokenizer, user_prompt, context_json, completion_json
    )

    assert 'user: Create a flowgraph with a source and a sink' in output
    assert '{ "flowgraph": { "nodes": [], "edges": [] } }' in output
    assert 'assistant: { "ok": 1 }' in output
    assert 'ADD_GEN:False' in output
