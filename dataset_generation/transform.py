#
# This file is part of the GNU Radio LLM project.
#

import json
import base64

from pathlib import Path

from dataset_generation.schema import Action

from dataset_generation.flowgraph import normalize_flowgraph_entry
from dataset_generation.runtime import normalize_runtime_entry

from flowgraph.schema import Flowgraph, FlowgraphAction


def encode_completion(graph: Flowgraph) -> str:
    graph_json = graph.model_dump_json().encode('utf-8')
    return base64.b64encode(graph_json).decode('utf-8')


def generate_prompt(action: Action) -> str:
    """
    Simple prompt generation until we can add some diversity.
    """
    match action:
        case _ if action.action == 'add_block':
            return f'Add a new block {action.block_id} to the flowgraph'
        case _ if action.action == 'remove_block':
            return f'Remove the block {action.block_id} from the flowgraph'
        case _ if action.action == 'connect':
            return f'Connect block {action.src} to block {action.dst}'
        case _ if action.action == 'disconnect':
            return f'Disconnect block {action.src} from block {action.dst}'
        case _ if action.action == 'parameter':
            return (f'Set the parameter {action.parameter} of block '
                    f'{action.block_id} to {action.value}')
        case _ if action.action == 'set':
            return (f'Set the {action.method} with the arguments {action.args} '
                    f'and {action.kwargs}')
        case _ if action.action == 'get':
            return (f'Get the {action.method} with the arguments {action.args} '
                    f'and {action.kwargs}')
        case _:
            return f'Perform the action {action.action}'


def build_dataset(trace_dir: Path, dataset_dir: Path):
    """
    Transform the traces into two datasets: runtime actions and flowgraph changes.
    """
    flowgraphs_dataset = []
    actions_dataset = []

    flowgraphs_dir = trace_dir / 'flowgraphs'
    actions_dir = trace_dir / 'actions'

    actions_dataset_path = dataset_dir / 'actions_dataset.jsonl'
    flowgraphs_dataset_path = dataset_dir / 'flowgraphs_dataset.jsonl'

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for trace_file in flowgraphs_dir.glob('*.jsonl'):
        history = []
        with trace_file.open('r') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    raise ValueError('Empty line in flowgraph trace file')

                entry = json.loads(line)
                actions = normalize_flowgraph_entry(line)
                snapshot = entry['snapshot_1']
                flowgraph = Flowgraph.model_validate(snapshot)
                for action in actions:
                    history.append({
                        'prompt': generate_prompt(action),
                        'completion': encode_completion(flowgraph)
                    })

        if history:
            flowgraphs_dataset.append(history)

    # TODO: Create the action dataset next

