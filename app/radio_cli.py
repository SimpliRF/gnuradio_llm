#!/usr/bin/env python3
#
# This file is part of the GNU Radio LLM project.
#

import sys

from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel

from prompt_toolkit import prompt

from llm.inference import ModelEngine
from flowgraph.schema import Flowgraph
from flowgraph.builder import FlowgraphBuilder
from flowgraph.controller import FlowgraphController


def draw_flowgraph_tree(console: Console, flowgraph: Flowgraph):
    """
    Draw a flowgraph tree in the console to display to the user.
    """
    tree = Tree(f'[bold green]{flowgraph.name}[/bold green]')

    block_map = {b.id: b for b in flowgraph.blocks}
    for src_id, dst_id in flowgraph.connections:
        src_block = block_map.get(src_id)
        dst_block = block_map.get(dst_id)
        if src_block and dst_block:
            node = tree.add(f'[cyan]{src_id}[/cyan] → [magenta]{dst_id}[/magenta]')
            node.add(f'{src_block.type} → {dst_block.type}')

    console.print(Panel(tree))


def draw_flowgraph_table(console: Console, flowgraph: Flowgraph):
    """
    Draw a flowgraph table in the console to display to the user.
    """
    block_table = Table(title='Blocks')
    block_table.add_column('ID', style='cyan', no_wrap=True)
    block_table.add_column('Name', style='blue')
    block_table.add_column('Type', style='green')
    block_table.add_column('Params', style='magenta')

    for block in flowgraph.blocks:
        block_table.add_row(
            block.id, block.name, block.type, str(block.parameters)
        )

    conn_table = Table(title='Connections')
    conn_table.add_column('Source ID', style='cyan')
    conn_table.add_column('Destination ID', style='cyan')

    for src_id, dst_id in flowgraph.connections:
        conn_table.add_row(src_id, dst_id)

    console.print(Panel(block_table))
    console.print(Panel(conn_table))


def main_entry():
    console = Console()

    console.print('[bold cyan] 🛰️  GNU Radio CLI Assistant[/bold cyan]')
    console.print('Type a description of a flowgraph you want to build.')
    console.print('Type [bold red]exit[/bold red] or [bold red]Ctrl+C[/bold red] to quit.')

    engine = ModelEngine()

    while True:
        try:
            user_input = prompt([('class:prompt', '» ')]).strip()
        except (KeyboardInterrupt, EOFError):
            print('\nExiting...')
            break

        if user_input.lower() in {'exit', 'quit'}:
            print('Exiting...')
            break

        if not user_input:
            continue

        try:
            flowgraph_json = engine.generate(user_input)
            console.print('[dim]Generated JSON:[/dim]')
            console.print_json(flowgraph_json)

            flowgraph = FlowgraphBuilder.from_json(flowgraph_json)
            console.print('[bold green]Flowgraph successfully built![/bold green]')

            controller = FlowgraphController(console)
            controller.load_flowgraph(flowgraph)
            console.print('[bold green]Flowgraph loaded successfully![/bold green]')

            # TODO: How to manage running the flowgraph?

        except Exception as e:
            console.print(f'[bold red]Error generating flowgraph:[/bold red] {e}')
            feedback = str(e)

            console.print('[yellow]Providing feedback to the LLM...[/yellow]')
            try:
                fixed_json = engine.retry_with_feedback(user_input, feedback)
                flowgraph = FlowgraphBuilder.from_json(fixed_json)
                controller.start(flowgraph)
                console.print('[green]✔ Retry successful![/green]')

                # TODO: How do we manage running the flowgraph concurrently?
            except Exception as retry_error:
                console.print(f'[bold red]❌ Retry failed:[/bold red] {retry_e}')

    return 0


if __name__ == '__main__':
    sys.exit(main_entry())
