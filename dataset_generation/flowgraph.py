#
# This file is part of the GNU Radio LLM project.
#

import json

from typing import Set, Tuple, List, Any

from datetime import datetime

from dataset_generation.schema import (
    AddBlockAction,
    RemoveBlockAction,
    ParameterAction,
    ConnectAction,
    DisconnectAction,
    Action
)


def snapshot_blocks(snapshot: dict[str, Any]) -> dict[str, Any]:
    blocks = {}
    for block in snapshot.get('blocks') or []:
        if isinstance(block, dict) and block.get('id'):
            blocks[str(block['id'])] = block
    return blocks


def snapshot_connections(snapshot: dict[str, Any]) -> Set[Tuple[str, str, str, str]]:
    connections = set()
    for conn in snapshot.get('connections') or []:
        if isinstance(conn, list) and len(conn) == 4:
            connections.add(tuple(conn))
    return connections


def flowgraph_diff(snapshot_0: dict[str, Any],
                   snapshot_1: dict[str, Any],
                   flowgraph_id: str,
                   timestamp: str) -> List[Action]:
    changes = []

    # Check for blocks that were added
    blocks_0 = snapshot_blocks(snapshot_0)
    blocks_1 = snapshot_blocks(snapshot_1)

    block_keys_0 = set(blocks_0.keys())
    block_keys_1 = set(blocks_1.keys())

    ts = datetime.fromisoformat(timestamp)
    for block_id in block_keys_1 - block_keys_0:
        block = blocks_1[block_id]
        changes.append(AddBlockAction(
            action='add_block',
            flowgraph_id=flowgraph_id,
            timestamp=ts,
            source='flowgraph',
            block_id=block_id,
            parameters=block.get('parameters', {}),
        ))

    # Check for blocks that were removed
    for block_id in block_keys_0 - block_keys_1:
        changes.append(RemoveBlockAction(
            flowgraph_id=flowgraph_id,
            timestamp=ts,
            source='flowgraph',
            block_id=block_id,
        ))

    # Check for parameter changes on blocks
    for block_id in block_keys_1 & block_keys_0:
        p_0 = blocks_0[block_id].get('parameters', {})
        p_1 = blocks_1[block_id].get('parameters', {})

        for k, v_1 in p_1.items():
            v_0 = p_0.get(k)
            if v_0 != v_1:
                changes.append(ParameterAction(
                    flowgraph_id=flowgraph_id,
                    timestamp=ts,
                    source='flowgraph',
                    block_id=block_id,
                    parameter=k,
                    value=v_1,
                ))

    # Check for connections that were added
    connections_0 = snapshot_connections(snapshot_0)
    connections_1 = snapshot_connections(snapshot_1)

    for (src, src_port, dst, dst_port) in connections_1 - connections_0:
        changes.append(ConnectAction(
            flowgraph_id=flowgraph_id,
            timestamp=ts,
            source='flowgraph',
            src=(src, src_port),
            dst=(dst, dst_port),
        ))

    # Check for connections that were removed
    for (src, src_port, dst, dst_port) in connections_0 - connections_1:
        changes.append(DisconnectAction(
            flowgraph_id=flowgraph_id,
            timestamp=ts,
            source='flowgraph',
            src=(src, src_port),
            dst=(dst, dst_port),
        ))
    return changes


def normalize_flowgraph_entry(entry_json: str) -> List[Action]:
    entry = json.loads(entry_json)
    flowgraph_id = entry['id']
    timestamp = entry['timestamp']
    snapshot_0 = entry.get('snapshot_0') or {'blocks': [], 'connections': []}
    snapshot_1 = entry['snapshot_1']
    return flowgraph_diff(snapshot_0, snapshot_1, flowgraph_id, timestamp)
