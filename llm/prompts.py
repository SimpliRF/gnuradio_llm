#
# This file is part of the GNU Radio LLM project.
#

from typing import Optional, List


SYSTEM_PROMPT_PREFIX = '''
You are an assistant that generates and controls GNU Radio flowgraphs.
Do not include any explanations or extra text. Return exactly ONE JSON object.
'''


def get_system_prompt() -> str:
    system_prompt = f'{SYSTEM_PROMPT_PREFIX}\n\n'
    return system_prompt


def build_prompt(tokenizer,
                 user_prompt: str,
                 completion_json: str = '',
                 flowgraph_json: Optional[str] = None,
                 generation_prompt: bool = True) -> str:
    """
    Build a consistent prompt for inference.
    """
    completion_json = completion_json.strip()
    system_prompt = get_system_prompt()
    messages = [
        {'role': 'system', 'content': system_prompt},
    ]

    if flowgraph_json:
        context_prompt = f'Here is the current flowgraph:\n{flowgraph_json}\n\n'
        messages.append({'role': 'system', 'content': context_prompt})

    messages.append({'role': 'user', 'content': user_prompt})

    if not generation_prompt:
        messages.append({'role': 'assistant', 'content': completion_json})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation_prompt
    )
    return prompt


def build_chained_prompt(tokenizer,
                         history: List[tuple[str, str, str]],
                         generation_prompt: bool = True) -> str:
    """
    Build a chained prompt from a sequence of (user_prompt, completion_json)
    pairs.
    """
    system_prompt = get_system_prompt()
    messages = [
        {'role': 'system', 'content': system_prompt}
    ]

    for user_prompt, context_json, completion_json in history[:-1]:
        completion_json = completion_json.strip()
        context_json = context_json.strip()
        if len(context_json) > 0:
            system_prompt = f'Here is the current flowgraph:\n{context_json}\n\n'
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})
        messages.append({'role': 'assistant', 'content': completion_json})

    context_json = history[-1][1].strip()
    if len(context_json) > 0:
        system_prompt = f'Here is the current flowgraph:\n{context_json}\n\n'
        messages.append({'role': 'system', 'content': system_prompt})

    user_prompt = history[-1][0]
    messages.append({'role': 'user', 'content': user_prompt})

    if not generation_prompt:
        completion_json = history[-1][1].strip()
        messages.append({'role': 'assistant', 'content': completion_json})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation_prompt
    )
    return prompt
