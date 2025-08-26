#
# This file is part of the GNU Radio LLM project.
#

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class Flowgraph(BaseModel):
    """
    Represents the data to be loaded by the GRC compiler.
    """
    options: Dict[str, Any] = Field(default_factory=dict)
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlowgraphAction(BaseModel):
    """
    Represents an action to be performed on a flowgraph.
    """
    action: str
    method: Optional[str] = None
    value: Optional[float | int | str | bool] = None


def minimize_flowgraph(flowgraph: Flowgraph) -> Flowgraph:
    """
    Minimize flowgraph by removing unnecessary data.
    """
    def remove_keys(data: Dict[str, Any], keys: tuple) -> Any:
        if isinstance(data, dict):
            return {k: remove_keys(v, keys) for k, v in data.items() if k not in keys}
        elif isinstance(data, list):
            return [remove_keys(item, keys) for item in data]
        else:
            return data

    keys_to_remove = (
        'comment',
        'coordinate',
        'copyright',
        'description',
        'category',
        'bus_sink',
        'bus_source',
        'bus_structure',
        'rotation',
        'alias',
        'affinity',
        'grc_version',
        'cmake_opt',
        'gen_cmake',
        'gen_linking',
        'placement',
        'qt_qss_theme',
        'window_size',
        'author',
        'sizing_mode',
        'realtime_scheduling',
        'bus_structure_sink',
        'run_options',
        'thread_safe_setters'
    )

    cleaned = remove_keys(flowgraph.model_dump(), keys_to_remove)
    return Flowgraph(**cleaned)
