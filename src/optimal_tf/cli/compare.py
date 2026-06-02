from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Sequence

from ..services import CompareRequest, run_compare
from ..rebalance import supported_rebalance_frequencies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare several optimal_tf strategies on the same evaluation run.')
    parser.add_argument('--config', type=str, default=str(Path('configs/optimal_tf.example.toml')), help='Path to a TOML config file.')
    parser.add_argument('--universe', type=str, default=None, help='Universe name from market_tickers_data.')
    parser.add_argument('--start', type=str, default=None, help='Start date for price history.')
    parser.add_argument('--strategies', type=str, default=None, help='Comma-separated list of strategies to compare.')
    parser.add_argument('--rebalance-frequency', type=str, default=None, choices=supported_rebalance_frequencies(), help='Portfolio rebalance schedule.')
    parser.add_argument('--evaluation-start', type=str, default=None, help='Start date for the evaluation window.')
    parser.add_argument('--evaluation-end', type=str, default=None, help='End date for the evaluation window.')
    parser.add_argument('--long-only', action=argparse.BooleanOptionalAction, default=None, help='Project weights to long-only.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory for comparison outputs.')
    parser.add_argument('--clean-output-dir', action=argparse.BooleanOptionalAction, default=True, help='Clean the output directory before writing comparison results.')
    parser.add_argument('--output-plot', action=argparse.BooleanOptionalAction, default=True, help='Generate comparison PNG charts.')
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    started_at = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    strategies = [] if args.strategies is None else [item.strip() for item in args.strategies.split(',') if item.strip()]
    result = run_compare(CompareRequest(
        config_path=args.config,
        universe=args.universe,
        start=args.start,
        strategies=strategies,
        rebalance_frequency=args.rebalance_frequency,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        long_only=args.long_only,
        output_dir=args.output_dir,
        clean_output_dir=args.clean_output_dir,
        output_plot=args.output_plot,
    ))
    print(f"universe: {result.universe}")
    print(f"strategies: {', '.join(result.strategies)}")
    print(f"execution_time_seconds: {perf_counter() - started_at: .3f}")
    if not result.comparison.summary_table.empty:
        print('summary:')
        print(result.comparison.summary_table.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
