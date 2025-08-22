#
# This file is part of the GNU Radio LLM project.
#

import json

from typing import List

from datetime import datetime

from dataset_generation.schema import (
    SetAction,
    GetAction,
    Action
)


def normalize_runtime_entry(entry_json: str) -> List[Action]:
    entry = json.loads(entry_json)
    method = entry['method']
    flowgraph_id = entry['flowgraph_id']
    timestamp = datetime.fromisoformat(entry['timestamp'])

    if method.startswith('set_'):
        return [SetAction(
            flowgraph_id=flowgraph_id,
            timestamp=timestamp,
            source='runtime',
            method=method,
            args=entry.get('args', []),
            kwargs=entry.get('kwargs', {}),
        )]

    return [GetAction(
        flowgraph_id=flowgraph_id,
        timestamp=timestamp,
        source='runtime',
        method=method,
        args=entry.get('args', []),
        kwargs=entry.get('kwargs', {}),
    )]
