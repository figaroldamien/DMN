from __future__ import annotations

import argparse
from pathlib import Path

from optimal_tf.services import VaryStrategyRequest, run_vary_strategy

DEFAULT_STRATEGIES = ("RP", "ARP", "NM")
DEFAULT_OUTPUT_DIR = "output/optimal_tf/evaluation/vary_strategy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the impact of varying the portfolio strategy.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    parser.add_argument("--rebalance-frequency", type=str, default=None)
    parser.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-periods-mode", choices=("clamp", "fixed"), default="clamp")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_vary_strategy(VaryStrategyRequest(
        config_path=args.config,
        universe=args.universe,
        start=args.start,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        rebalance_frequency=args.rebalance_frequency,
        strategies=[item.strip() for item in args.strategies.split(',') if item.strip()],
        method=args.method,
        window=args.window,
        matrix_date=args.matrix_date,
        output_dir=args.output_dir,
        min_periods_mode=args.min_periods_mode,
    ))
    print(f"universe: {result.universe}")
    print("scenario_summary:")
    print(result.scenario_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if result.highlights:
        print('scenario_highlights:')
        for key, value in result.highlights.items():
            print(f'- {key}: {value}')
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
