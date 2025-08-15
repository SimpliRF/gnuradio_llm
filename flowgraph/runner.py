#
# This file is part of the GNU Radio LLM project.
#

import importlib
import inspect
import pkgutil

from typing import Type, Dict
from gnuradio import gr
from flowgraph.schema import Flowgraph, Block


BLOCK_MODULES = [
    'gnuradio.blocks',
    'gnuradio.analog',
    'gnuradio.digital',
    'gnuradio.filter',
]


class FlowgraphRunner:
    def __init__(self, flowgraph: Flowgraph):
        self.flowgraph = flowgraph
        self.tb = gr.top_block()
        self.blocks: Dict[str, gr.basic_block] = {}
        self.block_registry = self._build_block_registry()

    def _build_block_registry(self) -> Dict[str, Type[gr.basic_block]]:
        registry = {}
        for module_name in BLOCK_MODULES:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, gr.basic_block)
                    and callable(obj)
                ):
                    registry[name] = obj
        return registry

    def _create_block(self, block: Block) -> gr.basic_block:
        if block.type not in self.block_registry:
            raise ValueError(f'Unknown block type: {block.type}')

        block_class = self.block_registry[block.type]
        try:
            return block_class(**block.parameters)
        except Exception as e:
            raise ValueError(f'Error creating block {block.name}: {e}')

    def _build(self):
        for block in self.flowgraph.blocks:
            self.blocks[block.id] = self._create_block(block)

        for conn in self.flowgraph.connections:
            src_id = conn['from'].split(':')[0]
            dst_id = conn['to'].split(':')[0]
            src_block = self.blocks.get(src_id)
            dst_block = self.blocks.get(dst_id)
            self.tb.connect(src_block, dst_block)

    def run(self):
        print('🔧  Building flowgraph...')
        self._build()

        print('🔁  Running flowgraph...')
        try:
            self.tb.run()
        except Exception as e:
            print(f'Error running flowgraph: {e}')

    def stop(self):
        print('⏹️  Stopping flowgraph...')
        self.tb.stop()
