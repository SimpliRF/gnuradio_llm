#
# This file is part of the GNU Radio LLM project.
#

from typing import Optional


SYSTEM_PROMPT_PREFIX = '''
You are an assistant that generates and controls GNU Radio flowgraphs.
Do not include any explanations or extra text. Return exactly ONE JSON object.
'''


def get_system_prompt() -> str:
    system_prompt = f'{SYSTEM_PROMPT_PREFIX}\n\n'
    return system_prompt


def build_prompt(tokenizer,
                 user_prompt: str,
                 context_json: Optional[str] = None,
                 completion_json: Optional[str] = None) -> str:
    """
    Build a consistent prompt for inference.
    """
    system_prompt = get_system_prompt()
    messages = []

    if context_json:
        system_prompt += f'Here is the current flowgraph:\n{context_json}\n\n'

    messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': user_prompt + '\n'})

    if completion_json:
        messages.append({'role': 'assistant', 'content': completion_json})

    add_generation_prompt = completion_json is None
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return prompt
