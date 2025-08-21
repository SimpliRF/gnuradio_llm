#!/usr/bin/env python3
#
# This file is part of the GNU Radio LLM project.
#

import sys
import argparse

from pathlib import Path

from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel

from prompt_toolkit import prompt

from pydantic import ValidationError

from flowgraph.schema import Flowgraph, FlowgraphAction
from flowgraph.controller import FlowgraphController

from llm.inference import ModelEngine
from llm.tune import ModelTrainer


def draw_flowgraph_tree(console: Console, flowgraph: Flowgraph):
    """
    Draw a flowgraph tree in the console to display to the user.
    """
    tree = Tree(f'[bold green]{flowgraph.name}[/bold green]')

    block_map = {b.id: b for b in flowgraph.blocks}
    for conn in flowgraph.connections:
        src_id = conn['from'].split(':')[0]
        dst_id = conn['to'].split(':')[0]
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
    block_table.add_column('ID', style='cyan')
    block_table.add_column('Name', style='green')
    block_table.add_column('Type', style='magenta')
    block_table.add_column('Params', style='magenta')

    for block in flowgraph.blocks:
        block_table.add_row(
            block.id, block.name, block.type, str(block.parameters)
        )

    conn_table = Table(title='Connections')
    conn_table.add_column('Source ID', style='cyan')
    conn_table.add_column('Destination ID', style='cyan')

    for conn in flowgraph.connections:
        conn_table.add_row(conn['from'], conn['to'])

    console.print(Panel(block_table))
    console.print(Panel(conn_table))


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='radio_cli',
        description='GNU Radio LLM - Inference and training CLI'
    )

    parser.add_argument(
        '--tree', action='store_true',
        help='Show flowgraph tree instead of table'
    )
    parser.add_argument(
        '--max-attempts', default=3, type=int,
        help='Maximum number of attempts for generating a valid response'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Run in training mode (LoRA on CPU, QLoRA on CUDA if available)'
    )
    parser.add_argument(
        '--model', default='Qwen/Qwen2.5-0.5B-Instruct', type=str,
        help='The model name to load (set to output for tuned model)'
    )
    parser.add_argument(
        '--dataset', default='dataset', type=Path,
        help='Directory containing training samples (*.json)'
    )
    parser.add_argument(
        '--output', default='output', type=Path,
        help='Directory to save model outputs'
    )
    return parser


def main_train(args: argparse.Namespace, console: Console) -> int:
    console.print('[bold yellow]Training mode activated...[/bold yellow]')

    if not args.dataset.exists():
        console.print('[bold red]❌ Dataset directory does not exist:[/bold red] {args.dataset}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer(dataset_dir=args.dataset, output_dir=args.output)
    trainer.train()

    console.print('[bold green]✔ Training complete![/bold green]')
    console.print(f'[dim]Model saved to: {args.output}[/dim]')
    return 0


def main_entry() -> int:
    parser = arg_parser()
    args = parser.parse_args()

    console = Console()
    if args.train:
        return main_train(args, console)

    console.print('[bold cyan] 🛰️  GNU Radio CLI Assistant[/bold cyan]')
    console.print('Type a description of a flowgraph you want to build.')
    console.print('Type [bold red]exit[/bold red] or [bold red]Ctrl+C[/bold red] to quit.')

    engine = ModelEngine(fallback_model_name=args.model)
    controller = FlowgraphController(console)

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

        response = engine.generate(user_input)
        console.print(f'[bold blue]LLM Response:[/bold blue]\n{response}')

        for attempt in range(args.max_attempts):
            try:
                # Try to parse the output as a flowgraph
                try:
                    flowgraph = Flowgraph.model_validate_json(response)

                    console.print('[bold green]✔ Flowgraph successfully built![/bold green]')
                    console.print('[dim]Generated JSON:[/dim]')
                    console.print_json(response)

                    controller.load_flowgraph(flowgraph)

                    if args.show_tree:
                        draw_flowgraph_tree(console, flowgraph)
                    else:
                        draw_flowgraph_table(console, flowgraph)

                    break
                except ValidationError:
                    console.print('[bold red]Must not be a flowgraph JSON[/bold red]')

                # Try to parse the output as a flowgraph action
                try:
                    action = FlowgraphAction.model_validate_json(response)
                    controller.handle_action(action)
                    console.print('[green]✔ Action executed[/green]')
                    break
                except ValidationError:
                    console.print(f'[bold red]Must not be a flowgraph action[/bold red]')

                raise ValueError('Invalid response fromat from LLM...')

            except Exception as e:
                console.print(f'[bold red]❌ Error processing response:[/bold red] {e}')
                if attempt < args.max_attempts - 1:
                    console.print('[yellow]Retrying with feedback...[/yellow]')
                    response = engine.retry_with_feedback(user_input, str(e))
                    console.print(f'[bold blue]LLM Response:[/bold blue]\n{response}')
                else:
                    console.print('[bold red]❌ Max attempts reached...[/bold red]')
    return 0


if __name__ == '__main__':
    sys.exit(main_entry())
