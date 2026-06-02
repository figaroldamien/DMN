from __future__ import annotations

import argparse
from pathlib import Path

from optimal_tf.services import VaryWindowRequest, run_vary_window

DEFAULT_WINDOWS = (40, 60, 80, 120, 252, 504, 1200)
DEFAULT_OUTPUT_DIR = "output/optimal_tf/evaluation/vary_window"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the impact of varying the covariance lookback window.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    parser.add_argument("--rebalance-frequency", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--windows", type=str, default=",".join(str(value) for value in DEFAULT_WINDOWS))
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-periods-mode", choices=("clamp", "fixed"), default="clamp")
    parser.add_argument("--log-scale", dest="log_scale", action="store_true")
    parser.add_argument("--linear-scale", dest="log_scale", action="store_false")
    parser.set_defaults(log_scale=True)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_vary_window(VaryWindowRequest(
        config_path=args.config,
        universe=args.universe,
        start=args.start,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        rebalance_frequency=args.rebalance_frequency,
        strategy=args.strategy,
        method=args.method,
        windows=[int(item.strip()) for item in args.windows.split(',') if item.strip()],
        matrix_date=args.matrix_date,
        output_dir=args.output_dir,
        min_periods_mode=args.min_periods_mode,
        log_scale=args.log_scale,
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
