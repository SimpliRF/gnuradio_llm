#!/usr/bin/env python3
#
# This file is part of the GNU Radio LLM project.
#

import sys
import argparse

from pathlib import Path

from rich.console import Console

from dataset_generation.transform import build_datasets


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='gen_dataset',
        description='Generate datasets from GRC logger trace data'
    )

    parser.add_argument(
        '--traces', default='traces', type=Path,
        help='Directory containing GRC trace data'
    )
    parser.add_argument(
        '--dataset', default='datasets', type=Path,
        help='Directory to save the generated dataset'
    )
    return parser


def main_entry():
    parser = arg_parser()
    args = parser.parse_args()

    console = Console()
    console.print('[bold yellow]🔄 Dataset generation activated...[/bold yellow]')

    build_datasets(args.traces, args.dataset)

    console.print('[bold green]✔ Dataset generation completed successfully![/bold green]')
    console.print('[dim]Generated dataset files:[/dim]')
    for file in args.dataset.glob('**/*.jsonl'):
        if file.is_file():
            console.print(f' - {args.dataset}/{file.relative_to(args.dataset)}')

    return 0


if __name__ == '__main__':
    sys.exit(main_entry())
