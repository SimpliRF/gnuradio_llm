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
                 completion_json: str = '',
                 flowgraph_json: Optional[str] = None,
                 generation_prompt: bool = True) -> str:
    """
    Build a consistent prompt for inference.
    """
    completion_json = completion_json.strip()
    system_prompt = get_system_prompt()
    if flowgraph_json:
        system_prompt += f'Here is the current flowgraph:\n{flowgraph_json}\n\n'
    else:
        system_prompt += 'Start with a new flowgraph with a variable sample rate block.\n\n'

    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

        if not generation_prompt:
            messages.append({'role': 'assistant', 'content': completion_json})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt
        )
        return prompt

    if generation_prompt:
        return (f'{system_prompt}\n\n### Prompt: '
                f'{user_prompt}\n\n### Completion: ')
    else:
        return (f'{system_prompt}\n\n### Prompt: '
                f'{user_prompt}\n\n### Completion: {completion_json}')


def build_chained_prompt(tokenizer,
                         history: list[tuple[str, str, str]],
                         generation_prompt: bool = True) -> str:
    """
    Build a chained prompt from a sequence of (user_prompt, completion_json)
    pairs.
    """
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {'role': 'system', 'content': get_system_prompt()}
        ]

        for user_prompt, context_json, completion_json in history[:-1]:
            completion_json = completion_json.strip()
            if len(context_json):
                system_prompt = f'Here is the current flowgraph:\n{context_json}\n\n'
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': user_prompt})
            messages.append({'role': 'assistant', 'content': completion_json})

        user_prompt = history[-1][0]
        messages.append({'role': 'user', 'content': user_prompt})

        if not generation_prompt:
            completion_json = history[-1][1].strip()
            messages.append({'role': 'assistant', 'content': completion_json})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt
        )
        return prompt

    system_prompt = get_system_prompt()
    result = system_prompt + '\n\n'
    for user_prompt, context_json, completion_json in history[:-1]:
        result += f'### Prompt: {user_prompt}\n\n'
        result += f'### Completion: {completion_json.strip()}\n\n'

    user_prompt = history[-1][0]
    result += f'### Prompt: {user_prompt}\n\n'

    if not generation_prompt:
        completion_json = history[-1][1].strip()
        result += f'### Completion: {completion_json}'

    return result


