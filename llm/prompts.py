#
# This file is part of the GNU Radio LLM project.
#

import json

from flowgraph.schema import Flowgraph


SYSTEM_PROMPT_PREFIX = '''
You are an assistant that generates GNU Radio flowgraphs as JSON output based
on user input.
You must respond only with JSON matching this schema:
'''


def get_system_prompt() -> str:
    json_data = Flowgraph.model_json_schema()
    json_str = json.dumps(json_data, indent=2)
    return f'{SYSTEM_PROMPT_PREFIX}\n{json_str}\n\n'

