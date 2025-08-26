# type: ignore
#
# This file is part of the GNU Radio LLM project.
#

import tempfile
import multiprocessing as mp

from multiprocessing import connection

from typing import Type
from pathlib import Path

from rich.console import Console

from gnuradio import gr
from gnuradio.grc.core.platform import Platform
from gnuradio.grc.core.generator.top_block import TopBlockGenerator

from flowgraph.loader import load_top_block
from flowgraph.schema import Flowgraph


class FlowgraphRunner:
    def __init__(self, flowgraph: Flowgraph, console: Console):
        self.flowgraph = flowgraph
        self.console = console
        self.process = None

        self.console.print('🔧 Preparing flowgraph...')

        self.generated_path = self._generate_flowgraph()

        self.tb_cls = self._create_top_block()
        self.tb = None
        if not self._is_gui():
            self.tb = self.tb_cls()

    def _generate_flowgraph(self) -> Path:
        platform = Platform(
            version=gr.version(),
            version_parts=(
                gr.major_version(),
                gr.api_version(),
                gr.minor_version()),
            prefs=gr.prefs(),
            install_prefix=gr.prefix()
        )
        platform.build_library()
        grc_flowgraph = platform.make_flow_graph()
        grc_flowgraph.import_data(self.flowgraph.model_dump())
        grc_flowgraph.rewrite()
        grc_flowgraph.validate()

        if not grc_flowgraph.is_valid():
            raise ValueError('Invalid flowgraph')

        generator = TopBlockGenerator(grc_flowgraph, tempfile.gettempdir())
        generator.write()
        return Path(generator.file_path)

    def _create_top_block(self) -> Type[gr.top_block]:
        _, top_block_cls = load_top_block(self.generated_path)
        return top_block_cls

    def _is_gui(self) -> bool:
        parameters = self.flowgraph.options['parameters']
        return parameters.get('generate_options') == 'qt_gui'

    def start(self):
        self.console.print('▶️ Starting flowgraph...')
        self.tb.start()

    def run(self):
        self.console.print('🔁 Running flowgraph...')
        try:
            self.tb.run()
        except Exception as e:
            self.console.print(f'Error running flowgraph: {e}')

    def stop(self):
        self.console.print('⏹️ Stopping flowgraph...')
        self.tb.stop()
