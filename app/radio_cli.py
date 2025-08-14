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

from flowgraph.schema import Flowgraph


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

        # TODO: Process user input and send it to the LLM

    return 0


if __name__ == '__main__':
    sys.exit(main_entry())
