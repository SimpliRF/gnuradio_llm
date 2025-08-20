#
# This file is part of the GNU Radio LLM project.
#

import functools

from grc_dataset_logger.logger import GRCLogger


GRC_FLOWGRAPH_METHODS = (
    'new_block',
    'remove_element',
    'connect',
    'disconnect',
)


GRC_BLOCK_METHODS = (
    'set_param',
)


def apply_patches(logger: GRCLogger):
    """
    This function patches selected methods on flowgraphs and blocks to avoid
    relying on Qt signals.
    """
    from gnuradio.grc.core.FlowGraph import FlowGraph
    from gnuradio.grc.core.blocks import Block
    from gnuradio.grc.core.Connection import Connection

    def wrap(cls, method, hook):
        original = getattr(cls, method)

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            try:
                hook(self, method, args, kwargs, result)
            except Exception as e:
                # Always avoid crashing GRC
                pass
            return result
        setattr(cls, method, wrapped)

    def on_flowgraph_change(self, method, args, kwargs, result):
        logger.on_flowgraph_change(self, method, args, kwargs)

    def on_block_param_change(self, method, args, kwargs, result):
        logger.on_block_param_change(self, method, args, kwargs)

    for method in GRC_FLOWGRAPH_METHODS:
        if hasattr(FlowGraph, method):
            wrap(FlowGraph, method, on_flowgraph_change)

    for method in GRC_BLOCK_METHODS:
        if hasattr(Block, method):
            wrap(Block, method, on_block_param_change)
