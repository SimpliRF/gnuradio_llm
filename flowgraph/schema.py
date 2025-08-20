#
# This file is part of the GNU Radio LLM project.
#

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class Block(BaseModel):
    """
    Represents a block in the flowgraph.
    """
    id: str
    name: str
    type: str
    parameters: Dict[str, float | int | str | bool] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)


class GUIConfig(BaseModel):
    """
    Represents the GUI configuration for a block.
    """
    enabled: bool = True
    visual_blocks: Optional[List[str]] = None


class MetaInfo(BaseModel):
    """
    Represents metadata information for the flowgraph.
    """
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class Flowgraph(BaseModel):
    """
    Represents a flowgraph containing multiple blocks.
    """
    name: Optional[str]
    blocks: List[Block] = Field(default_factory=list)
    connections: List[Dict[str, str]] = Field(default_factory=list)
    gui_config: GUIConfig = Field(default_factory=GUIConfig)
    meta_info: MetaInfo = Field(default_factory=MetaInfo)


class FlowgraphAction(BaseModel):
    """
    Represents an action to be performed on a flowgraph.
    """
    action: str
    block_id: Optional[str] = None
    method: Optional[str] = None
    value: Optional[float | int | str | bool] = None
