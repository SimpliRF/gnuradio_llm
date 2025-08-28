#
# This file is part of the GNU Radio LLM project.
#

import json
import base64
import functools

from typing import Type

from gnuradio.gr.top_block import top_block
from gnuradio.grc.core.FlowGraph import FlowGraph

from grc_dataset_logger.flowgraph_logger import FlowgraphLogger
from grc_dataset_logger.runtime_logger import RuntimeLogger


GRC_FLOWGRAPH_METHODS = (
    'get_run_command',
    'new_block',
    'remove_element',
    'connect',
    'disconnect',
)


def hook_method(cls, method, hook):
    original = getattr(cls, method)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            override = hook(self, method, args, kwargs, result)
            return override if override is not None else result
        except Exception as e:
            return result
    setattr(cls, method, wrapped)


def patch_flowgraph(logger: FlowgraphLogger):
    """
    This function patches selected methods in GRC's flowgraph model.
    """
    def on_flowgraph_change(flowgraph, method, args, kwargs, result):
        if method == 'get_run_command':
            if isinstance(result, str):
                script_path = result.split(' ')[-1]
                flowgraph_data = flowgraph.export_data()
                flowgraph_json = json.dumps(flowgraph_data)
                flowgraph_json = base64.b64encode(flowgraph_json.encode()).decode()
                wrapped = (f'python -c "from grc_dataset_logger.launch_top_block '
                           f'import execute_script; from collections '
                           f'import OrderedDict; '
                           f'execute_script(\'{script_path}\', '
                           f'\'{flowgraph_json}\')"')
                return wrapped
        logger.on_flowgraph_change(flowgraph, method, args, kwargs)
        return None

    for method in GRC_FLOWGRAPH_METHODS:
        if hasattr(FlowGraph, method):
            hook_method(FlowGraph, method, on_flowgraph_change)
    print('---> GRC dataset logger hooked GRC successfully')


def patch_top_block(tb_cls: Type[top_block], logger: RuntimeLogger):
    """
    This function patches setters and getters in a GRC generated top block.
    """
    def on_top_block_change(self, method, args, kwargs, result):
        logger.on_top_block_change(self, method, args, kwargs, result)
        return None

    for name in dir(tb_cls):
        if name.startswith('set_') or name.startswith('get_'):
            original = getattr(tb_cls, name, None)
            if callable(original):
                hook_method(tb_cls, name, on_top_block_change)
