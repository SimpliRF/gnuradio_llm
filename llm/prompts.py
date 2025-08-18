#
# This file is part of the GNU Radio LLM project.
#

import json

from flowgraph.schema import Flowgraph, FlowgraphAction


SYSTEM_PROMPT_PREFIX = '''
You are an assistant that generates GNU Radio flowgraphs.
You must respond only with JSON matching one of the following schemas:
- Flowgraph: when creating or modifying a flowgraph.
- FlowgraphAction: when performing control actions like start, stop, and more.

Do not include any explanations or extra text, only the JSON output.
'''


def get_system_prompt() -> str:
    flowgraph_schema = json.dumps(Flowgraph.model_json_schema(), indent=2)
    action_schema = json.dumps(FlowgraphAction.model_json_schema(), indent=2)
    return (f'{SYSTEM_PROMPT_PREFIX}\n\n'
            f'Flowgraph schema:\n{flowgraph_schema}\n\n'
            f'FlowgraphAction schema:\n{action_schema}\n\n')
