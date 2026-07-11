from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.data_quality import load_filtered_prices_for_universe
from optimal_tf.scripts.common import resolve_window_estimation_cfg
from optimal_tf.scripts.common.benchmark import matrix_sample_bundle
from optimal_tf.strategies.common import resolve_allocation_date
from trading_core.risk import clean_correlation_matrix, correlation_to_covariance

from .compare_cleaner_eigenvectors import (
    ESTIMATOR_METHOD_ALIASES,
    INPUT_TYPE_ALIASES,
    METHOD_ALIASES,
    _compare_vector_pair,
    _optimal_rank_matching,
    _ordered_eigenvectors,
    _resolve_requested_methods,
)

DEFAULT_SIZES = (50, 100, 200, 300, 400, 500)
DEFAULT_TOP_NS = (5, 10)
DEFAULT_METHODS = ("empirical", "linear_shrinkage", "rie_spectral")
DEFAULT_OUTPUT_DIR = "output/optimal_tf/inspection/sp500_cleaner_subsamples"


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare cleaner eigenvectors on random SP500 sub-universes.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--sizes", type=str, default=",".join(str(value) for value in DEFAULT_SIZES))
    parser.add_argument("--top-ns", type=str, default=",".join(str(value) for value in DEFAULT_TOP_NS))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--input-type", type=str, default="normalized_returns")
    parser.add_argument("--matrix-type", choices=("correlation", "covariance"), default="correlation")
    parser.add_argument("--estimator-method", choices=("sample_window", "ewma"), default="sample_window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-policy", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--disable-quality-filter", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sizes = _parse_int_csv(args.sizes)
    top_ns = _parse_int_csv(args.top_ns)
    requested_methods = _resolve_requested_methods(_parse_csv(args.methods))
    input_mode = INPUT_TYPE_ALIASES.get(args.input_type.strip().lower())
    if input_mode is None:
        raise ValueError(f"Unknown input type '{args.input_type}'.")
    estimator_method = ESTIMATOR_METHOD_ALIASES.get(args.estimator_method.strip().lower())
    if estimator_method is None:
        raise ValueError(f"Unknown estimator method '{args.estimator_method}'.")

    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(args.config)
    del backtest, allocation, compare, output
    universe = replace(universe, name="sp500", start=args.start or universe.start)
    evaluation_start = args.evaluation_start or evaluation.evaluation_start
    if args.disable_quality_filter:
        prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=args.refresh_policy)
        quality_summary = {
            "quality_filter_enabled": False,
            "quality_reference_start": evaluation_start,
            "kept_tickers": int(prices.shape[1]),
            "excluded_tickers": 0,
        }
    else:
        prices, quality_report = load_filtered_prices_for_universe(
            universe,
            evaluation_start=evaluation_start,
            refresh_policy=args.refresh_policy,
        )
        quality_summary = {
            "quality_filter_enabled": bool(quality_report.enabled),
            "quality_reference_start": quality_report.reference_start,
            "kept_tickers": int(len(quality_report.kept_tickers)),
            "excluded_tickers": int(len(quality_report.excluded_tickers)),
        }
    matrix_date = resolve_allocation_date(prices.index, as_of_date=args.matrix_date)
    history = prices.loc[prices.index <= matrix_date].copy()
    if history.empty:
        raise ValueError(f"No price history available on or before {matrix_date.date()} for sp500.")

    latest_row = history.ffill().iloc[-1]
    available_columns = [column for column in history.columns if pd.notna(latest_row.get(column))]
    max_size = max(sizes)
    if len(available_columns) < max_size:
        raise ValueError(f"Only {len(available_columns)} SP500 tickers available, cannot sample size {max_size}.")

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    matched_summary_rows: list[dict[str, object]] = []

    sampled_columns_by_size: dict[int, list[str]] = {}
    for size in sizes:
        sampled_columns_by_size[int(size)] = sorted(rng.choice(np.asarray(available_columns, dtype=object), size=int(size), replace=False).tolist())

    for size in sizes:
        sampled_columns = sampled_columns_by_size[int(size)]
        sampled_history = history.loc[:, sampled_columns].copy()
        estimator_window = max(2, 2 * int(size))
        window_estimation = resolve_window_estimation_cfg(estimation, estimator_window, min_periods_mode="clamp")
        empirical_corr, sample_cov, sample_size, sample_frame = matrix_sample_bundle(
            sampled_history,
            window_estimation,
            matrix_date,
            input_type=input_mode,
            estimator_method=estimator_method,
            estimator_window=estimator_window,
        )
        sample_vol = pd.Series(
            np.sqrt(np.clip(np.diag(sample_cov.to_numpy(dtype=float)), 0.0, None)),
            index=sample_cov.index,
            dtype=float,
        )

        cleaned_matrices: dict[str, pd.DataFrame] = {}
        for requested_name, resolved_name in requested_methods:
            cleaner_estimation = replace(estimation, cleaning_method=resolved_name)
            cleaned_corr = clean_correlation_matrix(
                empirical_corr,
                data=sample_frame,
                sample_size=sample_size,
                method=resolved_name,
                linear_shrinkage=cleaner_estimation.linear_shrinkage,
                bandwidth=cleaner_estimation.rie_bandwidth,
            )
            matrix = cleaned_corr
            if args.matrix_type == "covariance":
                matrix = correlation_to_covariance(cleaned_corr, sample_vol)
            cleaned_matrices[requested_name] = matrix

        for top_n in top_ns:
            method_vectors: dict[str, np.ndarray] = {}
            method_eigenvalues: dict[str, np.ndarray] = {}
            for requested_name, matrix in cleaned_matrices.items():
                eigenvalues, eigenvectors = _ordered_eigenvectors(matrix, top_n=top_n)
                method_vectors[requested_name] = eigenvectors
                method_eigenvalues[requested_name] = eigenvalues

            for left_name, right_name in combinations(method_vectors, 2):
                left_vectors = method_vectors[left_name]
                right_vectors = method_vectors[right_name]
                left_eigenvalues = method_eigenvalues[left_name]
                right_eigenvalues = method_eigenvalues[right_name]
                limit = min(left_vectors.shape[1], right_vectors.shape[1], int(top_n))

                pair_rows: list[dict[str, object]] = []
                for rank in range(limit):
                    abs_alignment, signed_alignment, max_abs_loading_diff, l2_loading_diff = _compare_vector_pair(
                        left_vectors[:, rank],
                        right_vectors[:, rank],
                    )
                    row = {
                        "universe": "sp500_sample",
                        "sample_size_target": int(size),
                        "top_n": int(top_n),
                        "matrix_date": matrix_date.strftime("%Y-%m-%d"),
                        "matrix_type": args.matrix_type,
                        "input_type": args.input_type,
                        "estimator_method": args.estimator_method,
                        "estimator_window": int(estimator_window),
                        "num_assets": int(empirical_corr.shape[0]),
                        "sample_size": int(sample_size),
                        "left_method": left_name,
                        "left_method_resolved": METHOD_ALIASES[left_name],
                        "right_method": right_name,
                        "right_method_resolved": METHOD_ALIASES[right_name],
                        "rank": int(rank + 1),
                        "left_eigenvalue": float(left_eigenvalues[rank]),
                        "right_eigenvalue": float(right_eigenvalues[rank]),
                        "abs_alignment": abs_alignment,
                        "signed_alignment": signed_alignment,
                        "max_abs_loading_diff": max_abs_loading_diff,
                        "l2_loading_diff": l2_loading_diff,
                    }
                    rank_rows.append(row)
                    pair_rows.append(row)

                matched_pair_rows: list[dict[str, object]] = []
                for left_rank, right_rank, matched_abs_alignment in _optimal_rank_matching(
                    left_vectors[:, :limit],
                    right_vectors[:, :limit],
                ):
                    abs_alignment, signed_alignment, max_abs_loading_diff, l2_loading_diff = _compare_vector_pair(
                        left_vectors[:, left_rank],
                        right_vectors[:, right_rank],
                    )
                    row = {
                        "universe": "sp500_sample",
                        "sample_size_target": int(size),
                        "top_n": int(top_n),
                        "matrix_date": matrix_date.strftime("%Y-%m-%d"),
                        "matrix_type": args.matrix_type,
                        "input_type": args.input_type,
                        "estimator_method": args.estimator_method,
                        "estimator_window": int(estimator_window),
                        "num_assets": int(empirical_corr.shape[0]),
                        "sample_size": int(sample_size),
                        "left_method": left_name,
                        "left_method_resolved": METHOD_ALIASES[left_name],
                        "right_method": right_name,
                        "right_method_resolved": METHOD_ALIASES[right_name],
                        "left_rank": int(left_rank + 1),
                        "right_rank": int(right_rank + 1),
                        "left_eigenvalue": float(left_eigenvalues[left_rank]),
                        "right_eigenvalue": float(right_eigenvalues[right_rank]),
                        "abs_alignment": abs_alignment,
                        "signed_alignment": signed_alignment,
                        "matched_abs_alignment": matched_abs_alignment,
                        "max_abs_loading_diff": max_abs_loading_diff,
                        "l2_loading_diff": l2_loading_diff,
                    }
                    matched_rows.append(row)
                    matched_pair_rows.append(row)

                matched_pair_frame = pd.DataFrame(matched_pair_rows)
                matched_summary_rows.append(
                    {
                        "universe": "sp500_sample",
                        "sample_size_target": int(size),
                        "top_n": int(top_n),
                        "matrix_date": matrix_date.strftime("%Y-%m-%d"),
                        "matrix_type": args.matrix_type,
                        "input_type": args.input_type,
                        "estimator_method": args.estimator_method,
                        "estimator_window": int(estimator_window),
                        "num_assets": int(empirical_corr.shape[0]),
                        "sample_size": int(sample_size),
                        "left_method": left_name,
                        "left_method_resolved": METHOD_ALIASES[left_name],
                        "right_method": right_name,
                        "right_method_resolved": METHOD_ALIASES[right_name],
                        "num_ranks": int(len(matched_pair_frame)),
                        "mean_abs_alignment": float(matched_pair_frame["abs_alignment"].mean()),
                        "min_abs_alignment": float(matched_pair_frame["abs_alignment"].min()),
                        "max_abs_loading_diff": float(matched_pair_frame["max_abs_loading_diff"].max()),
                        "mean_l2_loading_diff": float(matched_pair_frame["l2_loading_diff"].mean()),
                        "num_ranks_below_0p99": int((matched_pair_frame["abs_alignment"] < 0.99).sum()),
                        "num_ranks_below_0p95": int((matched_pair_frame["abs_alignment"] < 0.95).sum()),
                        "num_ranks_below_0p90": int((matched_pair_frame["abs_alignment"] < 0.90).sum()),
                        "rank_mapping": ",".join(
                            f"{int(row['left_rank'])}->{int(row['right_rank'])}" for _, row in matched_pair_frame.iterrows()
                        ),
                        "sampled_tickers": ",".join(sampled_columns),
                    }
                )

    rank_frame = pd.DataFrame(rank_rows).sort_values(
        ["sample_size_target", "top_n", "left_method", "right_method", "rank"],
        ignore_index=True,
    )
    matched_frame = pd.DataFrame(matched_rows).sort_values(
        ["sample_size_target", "top_n", "left_method", "right_method", "left_rank"],
        ignore_index=True,
    )
    matched_summary_frame = pd.DataFrame(matched_summary_rows).sort_values(
        ["sample_size_target", "top_n", "left_method", "right_method"],
        ignore_index=True,
    )

    rank_path = output_dir / "sp500_sample_alignment.csv"
    matched_path = output_dir / "sp500_sample_alignment_matched.csv"
    matched_summary_path = output_dir / "sp500_sample_alignment_matched_summary.csv"
    quality_path = output_dir / "quality_filter_summary.csv"
    rank_frame.to_csv(rank_path, index=False)
    matched_frame.to_csv(matched_path, index=False)
    matched_summary_frame.to_csv(matched_summary_path, index=False)
    pd.DataFrame([quality_summary]).to_csv(quality_path, index=False)

    print(f"sizes: {', '.join(str(value) for value in sizes)}")
    print(f"top_ns: {', '.join(str(value) for value in top_ns)}")
    print(f"seed: {int(args.seed)}")
    print(f"quality_filter_enabled: {not args.disable_quality_filter}")
    print(f"evaluation_start: {evaluation_start or 'config/default'}")
    print(f"matrix_type: {args.matrix_type}")
    print(f"input_type: {args.input_type}")
    print(f"estimator_method: {args.estimator_method}")
    print(f"rank_alignment_csv: {rank_path}")
    print(f"matched_alignment_csv: {matched_path}")
    print(f"matched_summary_csv: {matched_summary_path}")
    print(f"quality_summary_csv: {quality_path}")
    print("matched_summary_preview:")
    print(
        matched_summary_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
