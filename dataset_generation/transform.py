#
# This file is part of the GNU Radio LLM project.
#

import json
import base64

from pathlib import Path
from pydantic import BaseModel

from dataset_generation.schema import Action
from dataset_generation.flowgraph import normalize_flowgraph_entry
from dataset_generation.runtime import normalize_runtime_entry

from flowgraph.schema import Flowgraph, minimize_flowgraph


def encode_completion(data: BaseModel) -> str:
    data_json = data.model_dump_json().encode('utf-8')
    return base64.b64encode(data_json).decode('utf-8')


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
            return (f'Connect the block {action.src[0]} at '
                    f'port {action.src[1]} to {action.dst[0]} '
                    f'at port {action.dst[1]}')
        case _ if action.action == 'disconnect':
            return (f'Disconnect the block {action.src[0]} at '
                    f'port {action.src[1]} to {action.dst[0]} '
                    f'at port {action.dst[1]}')
        case _ if action.action == 'parameter':
            return (f'Set the parameter {action.parameter} of block '
                    f'{action.block_id} to {action.value}')
        case _ if action.action == 'set':
            return (f'Set the {action.method} method with the '
                    f'positional arguments: {action.args}, '
                    f'and keyword arguments: {action.kwargs}')
        case _ if action.action == 'get':
            return (f'Get the {action.method} method with the '
                    f'positional arguments: {action.args}, '
                    f'and keyword arguments: {action.kwargs}')
        case _:
            return f'Perform the action {action.action}'


def build_datasets(trace_dir: Path, dataset_dir: Path):
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
                flowgraph = Flowgraph(**entry['snapshot_1'])
                flowgraph = minimize_flowgraph(flowgraph)

                for action in actions:
                    history.append({
                        'prompt': generate_prompt(action),
                        'completion': encode_completion(flowgraph)
                    })

        if history:
            flowgraphs_dataset.append(history)

    for trace_file in actions_dir.glob('*.jsonl'):
        history = []
        with trace_file.open('r') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    raise ValueError('Empty line in actions trace file')

                entry = json.loads(line)
                actions = normalize_runtime_entry(line)
                for action in actions:
                    history.append({
                        'prompt': generate_prompt(action),
                        'completion': encode_completion(action)
                    })
        if history:
            actions_dataset.append(history)

    if flowgraphs_dataset:
        with flowgraphs_dataset_path.open('w') as fp:
            for history in flowgraphs_dataset:
                json.dump(history, fp)
                fp.write('\n')

    if actions_dataset:
        with actions_dataset_path.open('w') as fp:
            for history in actions_dataset:
                json.dump(history, fp)
                fp.write('\n')
