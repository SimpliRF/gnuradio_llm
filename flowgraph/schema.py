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
