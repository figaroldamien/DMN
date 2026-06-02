from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Sequence

import pandas as pd

from ..allocation import supported_strategies
from ..services import AllocationRequest, run_allocation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute optimal_tf portfolio weights on a given allocation date.")
    parser.add_argument('--config', type=str, default=str(Path('configs/optimal_tf.example.toml')), help='Path to a TOML config file.')
    parser.add_argument('--universe', type=str, default=None, help='Universe name from market_tickers_data.')
    parser.add_argument('--start', type=str, default=None, help='Start date for price history.')
    parser.add_argument('--date', type=str, default=None, help='Allocation date. Defaults to today.')
    parser.add_argument('--strategy', type=str, default=None, choices=supported_strategies(), help='Portfolio recipe to use.')
    parser.add_argument('--long-only', action=argparse.BooleanOptionalAction, default=None, help='Project weights to long-only after the portfolio recipe.')
    parser.add_argument('--output-csv', type=str, default=None, help='Optional path to save the weights as CSV.')
    parser.add_argument('--output-json', type=str, default=None, help='Optional path to save the weights as JSON.')
    return parser


def _format_weights(weights: pd.Series) -> str:
    display = weights[weights != 0.0].sort_values(ascending=False)
    if display.empty:
        display = weights.sort_values(ascending=False)
    return display.to_string(float_format=lambda x: f"{x: .6f}")


def run(argv: Sequence[str] | None = None) -> int:
    started_at = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = None
    if args.output_csv or args.output_json:
        output_dir = str(Path(args.output_csv or args.output_json).parent)
    result = run_allocation(AllocationRequest(
        config_path=args.config,
        universe=args.universe,
        start=args.start,
        as_of_date=args.date,
        strategy=args.strategy,
        long_only=args.long_only,
        output_dir=output_dir,
    ))
    print(f"strategy: {result.strategy}")
    print(f"universe: {result.universe}")
    print(f"requested_date: {pd.Timestamp(args.date).date() if args.date else pd.Timestamp.today().date()}")
    print(f"allocation_date: {result.allocation_date.date()}")
    print(f"signal_scale: {result.signal_scale: .6f}")
    print(f"num_assets: {(result.weights != 0.0).sum()}")
    print(f"execution_time_seconds: {perf_counter() - started_at: .3f}")
    print(_format_weights(result.weights))
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
