from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Sequence

from ..allocation import supported_strategies
from ..rebalance import supported_rebalance_frequencies
from ..services import StandardEvaluationRequest, run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a periodic evaluation of optimal_tf portfolios.')
    parser.add_argument('--config', type=str, default=str(Path('configs/optimal_tf.example.toml')), help='Path to a TOML config file.')
    parser.add_argument('--universe', type=str, default=None, help='Universe name from market_tickers_data.')
    parser.add_argument('--start', type=str, default=None, help='Start date for price history.')
    parser.add_argument('--strategy', type=str, default=None, choices=supported_strategies(), help='Portfolio recipe to use.')
    parser.add_argument('--rebalance-frequency', type=str, default=None, choices=supported_rebalance_frequencies(), help='Portfolio rebalance schedule.')
    parser.add_argument('--evaluation-start', type=str, default=None, help='Start date for the evaluation window.')
    parser.add_argument('--evaluation-end', type=str, default=None, help='End date for the evaluation window.')
    parser.add_argument('--long-only', action=argparse.BooleanOptionalAction, default=None, help='Project weights to long-only.')
    parser.add_argument('--output-dir', type=str, default=None, help='Optional directory for CSV/JSON exports.')
    parser.add_argument('--output-plot', action=argparse.BooleanOptionalAction, default=True, help='Generate a PNG performance chart when --output-dir is provided.')
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    started_at = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_evaluation(StandardEvaluationRequest(
        config_path=args.config,
        universe=args.universe,
        start=args.start,
        strategy=args.strategy,
        rebalance_frequency=args.rebalance_frequency,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        long_only=args.long_only,
        output_dir=args.output_dir,
        output_plot=args.output_plot,
    ))
    print(f"strategy: {result.strategy}")
    print(f"universe: {result.universe}")
    print(f"execution_time_seconds: {perf_counter() - started_at: .3f}")
    for key, value in asdict(result.evaluation_result.summary).items():
        print(f"{key}: {value}")
    if 'plot' in result.artifacts.files:
        print(f"plot: {result.artifacts.files['plot']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
