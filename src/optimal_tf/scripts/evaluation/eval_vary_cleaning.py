from __future__ import annotations

import argparse
from pathlib import Path

from optimal_tf.services import VaryCleaningRequest, run_vary_cleaning

DEFAULT_METHODS = ("empirical", "linear_shrinkage", "rie", "rie_reference")
DEFAULT_OUTPUT_DIR = "output/optimal_tf/evaluation/vary_cleaning"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the impact of varying the correlation cleaning method.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    parser.add_argument("--rebalance-frequency", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_vary_cleaning(VaryCleaningRequest(
        config_path=args.config,
        universe=args.universe,
        start=args.start,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        rebalance_frequency=args.rebalance_frequency,
        strategy=args.strategy,
        methods=[item.strip() for item in args.methods.split(',') if item.strip()],
        window=args.window,
        matrix_date=args.matrix_date,
        output_dir=args.output_dir,
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
