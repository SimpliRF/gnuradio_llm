#
# This file is part of the GNU Radio LLM project.
#

import importlib
import inspect
import pkgutil

from typing import Any, Dict
from gnuradio import gr
from flowgraph.schema import Flowgraph, Block


BLOCK_MODULES = [
    'gnuradio.blocks',
    'gnuradio.analog',
    'gnuradio.digital',
    'gnuradio.filter',
]


BLOCK_BASE_CLASSES = (
    gr.basic_block,
    gr.sync_block,
    gr.decim_block,
    gr.interp_block,
    gr.hier_block2
)


class FlowgraphRunner:
    def __init__(self, flowgraph: Flowgraph):
        self.flowgraph = flowgraph
        self.tb = gr.top_block()
        self.blocks: Dict[str, Any] = {}
        self.block_registry = self._build_block_registry()

    @staticmethod
    def _is_gr_block(obj: Any) -> bool:
        try:
            if isinstance(obj, BLOCK_BASE_CLASSES):
                return True
        except:
            pass

        if inspect.isclass(obj):
            return all(
                callable(getattr(obj, name, None))
                for name in ('input_signature', 'output_signature')
            )
        return False

    def _build_block_registry(self) -> Dict[str, Any]:
        registry = {}
        for module_name in BLOCK_MODULES:
            root = importlib.import_module(module_name)

            for _, mod_name, _ in pkgutil.walk_packages(
                root.__path__, prefix=root.__name__ + '.'
            ):
                try:
                    module = importlib.import_module(mod_name)
                    for name, obj in inspect.getmembers(module):
                        if not inspect.isclass(obj):
                            continue

                        is_block = False
                        if (issubclass(obj, BLOCK_BASE_CLASSES)
                            and obj not in BLOCK_BASE_CLASSES
                            or self._is_gr_block(obj)
                        ):
                            is_block = True

                        if is_block:
                            registry[name] = obj
                except Exception as e:
                    print(f'Error loading module {mod_name}: {e}')
        return registry

    def _create_block(self, block: Block) -> Any:
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
