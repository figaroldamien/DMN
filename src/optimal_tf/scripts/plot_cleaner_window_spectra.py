from __future__ import annotations

import argparse
from pathlib import Path

from optimal_tf.services import EigenvectorInspectionRequest, run_eigenvector_inspection

DEFAULT_WINDOWS = (40, 60, 80, 120, 252, 504, 1200)
DEFAULT_METHOD = "rie_reference"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare one cleaner spectrum across multiple covariance windows.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    parser.add_argument("--rebalance-frequency", type=str, default=None)
    parser.add_argument("--windows", type=str, default=",".join(str(value) for value in DEFAULT_WINDOWS))
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output/optimal_tf/inspection/eigenvectors")
    parser.add_argument("--min-periods-mode", choices=("clamp", "fixed"), default="clamp")
    parser.add_argument("--selection-mode", choices=("mp", "cumulative_variance", "top_n"), default="mp")
    parser.add_argument("--selection-cumulative-variance", type=float, default=80.0)
    parser.add_argument("--selection-top-n", type=int, default=3)
    parser.add_argument("--log-scale", dest="log_scale", action="store_true")
    parser.add_argument("--linear-scale", dest="log_scale", action="store_false")
    parser.set_defaults(log_scale=True)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_eigenvector_inspection(EigenvectorInspectionRequest(
        config_path=args.config,
        universe=args.universe,
        start=args.start,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        rebalance_frequency=args.rebalance_frequency,
        method=args.method,
        windows=[int(item.strip()) for item in args.windows.split(',') if item.strip()],
        matrix_date=args.matrix_date,
        output_dir=args.output_dir,
        min_periods_mode=args.min_periods_mode,
        selection_mode=args.selection_mode,
        selection_cumulative_variance=args.selection_cumulative_variance,
        selection_top_n=args.selection_top_n,
        log_scale=args.log_scale,
    ))
    print(f"universe: {result.universe}")
    print(f"matrix_date: {result.matrix_date.date()}")
    print(f"method: {result.method}")
    print(f"windows: {', '.join(str(window) for window in result.windows)}")
    print(f"sector_presence_shape: {result.sector_presence.shape}")
    print(f"loadings_shape: {result.loadings.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
