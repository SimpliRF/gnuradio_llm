#!/usr/bin/env python3
#
# This file is part of the GNU Radio LLM project.
#

import sys
import argparse

from pathlib import Path

from rich.console import Console

from llm.tune import ModelTrainer


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='radio_cli',
        description='GNU Radio LLM - Inference and training CLI'
    )

    parser.add_argument(
        '--model', default='Qwen/Qwen2-1.5B-Chat', type=str,
        help='The model name to load (set to output for tuned model)'
    )
    parser.add_argument(
        '--dataset', default='datasets', type=Path,
        help='Directory containing training samples (*.json)'
    )
    parser.add_argument(
        '--output', default='output', type=Path,
        help='Directory to save model outputs'
    )
    return parser


def main_entry():
    parser = arg_parser()
    args = parser.parse_args()

    console = Console()
    console.print('[bold yellow]🔄 Training activated...[/bold yellow]')

    if not args.dataset.exists():
        console.print('[bold red]❌ Dataset directory does not exist:[/bold red] {args.dataset}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    trainer = ModelTrainer(
        dataset_dir=args.dataset,
        model_name=args.model,
        output_dir=args.output
    )
    trainer.train()

    console.print('[bold green]✔ Training complete![/bold green]')
    console.print(f'[dim]Model saved to: {args.output}[/dim]')
    return 0


if __name__ == '__main__':
    sys.exit(main_entry())
