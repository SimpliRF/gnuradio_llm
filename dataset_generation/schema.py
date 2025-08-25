#
# This file is part of the GNU Radio LLM project.
#

from typing import Literal, List, Dict, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BaseAction(BaseModel):
    action: str
    timestamp: datetime
    flowgraph_id: str
    source: Literal['flowgraph', 'runtime']


class AddBlockAction(BaseAction):
    action: Literal['add_block'] = Field(default='add_block')
    block_id: str
    parameters: dict[str, Any]


class RemoveBlockAction(BaseAction):
    action: Literal['remove_block'] = Field(default='remove_block')
    block_id: str


class ParameterAction(BaseAction):
    action: Literal['parameter'] = Field(default='parameter')
    block_id: str
    parameter: str
    value: Any


class ConnectAction(BaseAction):
    action: Literal['connect'] = Field(default='connect')
    src: tuple[str, str]
    dst: tuple[str, str]


class DisconnectAction(BaseAction):
    action: Literal['disconnect'] = Field(default='disconnect')
    src: tuple[str, str]
    dst: tuple[str, str]


class RuntimeAction(BaseAction):
    method: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


class SetAction(RuntimeAction):
    action: Literal['set'] = Field(default='set')


class GetAction(RuntimeAction):
    action: Literal['get'] = Field(default='get')


Action = Union[
    AddBlockAction,
    RemoveBlockAction,
    ConnectAction,
    DisconnectAction,
    ParameterAction,
    SetAction,
    GetAction
]
