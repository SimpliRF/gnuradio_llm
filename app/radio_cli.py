#!/usr/bin/env python3
#
# This file is part of the GNU Radio LLM project.
#

import sys
import argparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from prompt_toolkit import prompt

from pydantic import ValidationError

from flowgraph.schema import Flowgraph, FlowgraphAction
from flowgraph.controller import FlowgraphController

from llm.inference import ModelEngine


def draw_flowgraph_table(console: Console, flowgraph: Flowgraph):
    """
    Draw a flowgraph table in the console to display to the user.
    """
    block_table = Table(title='Blocks')
    block_table.add_column('ID', style='cyan')
    block_table.add_column('Name', style='magenta')

    for block in flowgraph.blocks:
        block_table.add_row(block['id'], block['name'])

    conn_table = Table(title='Connections')
    conn_table.add_column('Source ID', style='cyan')
    conn_table.add_column('Destination ID', style='cyan')

    for conn in flowgraph.connections:
        conn_table.add_row(conn[0], conn[2])

    console.print(Panel(block_table))
    console.print(Panel(conn_table))


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='radio_cli',
        description='GNU Radio LLM - Inference and training CLI'
    )

    parser.add_argument(
        '--max-attempts', default=3, type=int,
        help='Maximum number of attempts for generating a valid response'
    )
    parser.add_argument(
        '--model', default='output', type=str,
        help='The model name to load (default is the tuned output model)'
    )
    return parser


def main_entry() -> int:
    parser = arg_parser()
    args = parser.parse_args()

    console = Console()

    console.print('[bold cyan] 🛰️  GNU Radio CLI Assistant[/bold cyan]')
    console.print('Type a description of a flowgraph you want to build.')
    console.print('Type [bold red]exit[/bold red] or [bold red]Ctrl+C[/bold red] to quit.')

    engine = ModelEngine(model_name=args.model)
    controller = FlowgraphController(console)

    current_flowgraph = None

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

        response = engine.generate(user_input, current_flowgraph)
        console.print(f'[bold blue]LLM Response:[/bold blue]\n{response}')

        for attempt in range(args.max_attempts):
            try:
                # Try to parse the output as a flowgraph
                try:
                    flowgraph = Flowgraph.model_validate_json(response)
                    console.print('[dim]Generated JSON:[/dim]')
                    console.print_json(response)

                    controller.load_flowgraph(flowgraph)
                    current_flowgraph = response

                    console.print('[bold green]✔ Flowgraph successfully built![/bold green]')

                    draw_flowgraph_table(console, flowgraph)

                    break
                except ValidationError:
                    console.print('[bold yellow]Must not be a flowgraph JSON[/bold yellow]')

                # Try to parse the output as a flowgraph action
                try:
                    action = FlowgraphAction.model_validate_json(response)
                    console.print('[dim]Generated JSON:[/dim]')
                    console.print_json(response)

                    controller.handle_action(action)

                    console.print('[green]✔ Action executed[/green]')
                    break
                except ValidationError:
                    pass

                raise ValueError('Invalid response format from LLM...')

            except Exception as e:
                console.print(f'[bold red]❌ Error processing response:[/bold red] {e}')
                if attempt < args.max_attempts - 1:
                    console.print('[yellow]Retrying with feedback...[/yellow]')
                    response = engine.retry_with_feedback(
                        user_input, str(e), current_flowgraph
                    )
                    console.print(f'[bold blue]LLM Response:[/bold blue]\n{response}')
                else:
                    console.print('[bold red]❌ Max attempts reached...[/bold red]')
    return 0


if __name__ == '__main__':
    sys.exit(main_entry())
