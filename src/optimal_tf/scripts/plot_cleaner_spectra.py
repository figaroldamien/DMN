from __future__ import annotations

import argparse
from pathlib import Path

from optimal_tf.services import SpectrumByCleanerRequest, run_spectrum_by_cleaner

DEFAULT_METHODS = ("empirical", "linear_shrinkage", "rie_spectral", "rie_reference")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate scree plots for selected correlation cleaning methods.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    parser.add_argument("--rebalance-frequency", type=str, default=None)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output/optimal_tf/spectral/by_cleaner")
    parser.add_argument("--log-scale", dest="log_scale", action="store_true")
    parser.add_argument("--linear-scale", dest="log_scale", action="store_false")
    parser.set_defaults(log_scale=True)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_spectrum_by_cleaner(SpectrumByCleanerRequest(
        config_path=args.config,
        universe=args.universe,
        start=args.start,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        rebalance_frequency=args.rebalance_frequency,
        methods=[item.strip() for item in args.methods.split(',') if item.strip()],
        matrix_date=args.matrix_date,
        output_dir=args.output_dir,
        log_scale=args.log_scale,
    ))
    print(f"universe: {result.universe}")
    print(f"matrix_date: {result.matrix_date.date()}")
    print(f"methods: {', '.join(result.methods)}")
    print(f"num_assets: {result.num_assets}")
    print(f"sample_size: {result.sample_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
