from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.data_quality import load_filtered_prices_for_universe
from optimal_tf.scripts.common import merge_common_overrides, resolve_window_estimation_cfg
from optimal_tf.scripts.common.benchmark import matrix_sample_bundle
from optimal_tf.services.inspection import UNIVERSE_COMPONENTS
from optimal_tf.strategies.common import resolve_allocation_date
from trading_core.risk import clean_correlation_matrix, correlation_to_covariance

DEFAULT_METHODS = ("empirical", "linear_shrinkage", "rie_spectral")
DEFAULT_OUTPUT_DIR = "output/optimal_tf/inspection/cleaner_eigenvector_alignment"
DEFAULT_TOP_N = 10

METHOD_ALIASES = {
    "empirical": "empirical",
    "linear_shrinkage": "linear_shrinkage",
    "rie": "rie_spectral",
    "rie_spectral": "rie_spectral",
    "rie_reference": "rie_reference",
}
INPUT_TYPE_ALIASES = {
    "normalized": "normalized",
    "normalized_returns": "normalized",
    "raw": "raw",
    "raw_returns": "raw",
}
ESTIMATOR_METHOD_ALIASES = {
    "sample_window": "window_sample",
    "window_sample": "window_sample",
    "ewma": "ewma_cross",
    "ewma_cross": "ewma_cross",
}
DEFAULT_UNIVERSES = tuple(sorted(name for name in UNIVERSE_COMPONENTS if name != "table8_all"))


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_requested_methods(raw_methods: list[str]) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    for raw_method in raw_methods:
        key = raw_method.strip().lower()
        method = METHOD_ALIASES.get(key)
        if method is None:
            raise ValueError(f"Unknown cleaning method '{raw_method}'. Allowed values: {sorted(METHOD_ALIASES)}")
        resolved.append((key, method))
    if len(resolved) < 2:
        raise ValueError("At least two cleaning methods must be provided.")
    return resolved


def _resolve_universes(raw_universes: list[str]) -> list[str]:
    universes = raw_universes or list(DEFAULT_UNIVERSES)
    invalid = [name for name in universes if name not in UNIVERSE_COMPONENTS]
    if invalid:
        raise ValueError(f"Unknown universes {invalid}. Allowed values: {sorted(UNIVERSE_COMPONENTS)}")
    return universes


def _ordered_eigenvectors(matrix: pd.DataFrame, *, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.to_numpy(dtype=float))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order].astype(float)
    eigenvectors = eigenvectors[:, order].astype(float)
    limit = min(max(1, int(top_n)), eigenvectors.shape[1])
    return eigenvalues[:limit], eigenvectors[:, :limit]


def _compare_vector_pair(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    signed_alignment = float(np.dot(left, right))
    abs_alignment = float(abs(signed_alignment))
    oriented_right = right if signed_alignment >= 0.0 else -right
    max_abs_loading_diff = float(np.max(np.abs(left - oriented_right)))
    l2_loading_diff = float(np.linalg.norm(left - oriented_right))
    return abs_alignment, signed_alignment, max_abs_loading_diff, l2_loading_diff


def _optimal_rank_matching(left_vectors: np.ndarray, right_vectors: np.ndarray) -> list[tuple[int, int, float]]:
    overlap = np.abs(left_vectors.T @ right_vectors)
    left_idx, right_idx = linear_sum_assignment(-overlap)
    pairs = [
        (int(left_rank), int(right_rank), float(overlap[left_rank, right_rank]))
        for left_rank, right_rank in zip(left_idx, right_idx, strict=True)
    ]
    pairs.sort(key=lambda item: item[0])
    return pairs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare leading cleaner eigenvectors across universes.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universes", type=str, default=",".join(DEFAULT_UNIVERSES))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--input-type", type=str, default="normalized_returns")
    parser.add_argument("--matrix-type", choices=("correlation", "covariance"), default="correlation")
    parser.add_argument("--estimator-method", choices=("sample_window", "ewma"), default="sample_window")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-policy", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--disable-quality-filter", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    universes = _resolve_universes(_parse_csv(args.universes))
    requested_methods = _resolve_requested_methods(_parse_csv(args.methods))
    input_mode = INPUT_TYPE_ALIASES.get(args.input_type.strip().lower())
    if input_mode is None:
        raise ValueError(f"Unknown input type '{args.input_type}'.")
    estimator_method = ESTIMATOR_METHOD_ALIASES.get(args.estimator_method.strip().lower())
    if estimator_method is None:
        raise ValueError(f"Unknown estimator method '{args.estimator_method}'.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    matched_alignment_rows: list[dict[str, object]] = []
    matched_summary_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    for universe_name in universes:
        universe, estimation, backtest, allocation, evaluation, compare, output = load_config(args.config)
        del allocation, compare, output
        universe, estimation, backtest, evaluation = merge_common_overrides(
            universe,
            estimation,
            backtest,
            evaluation,
            type(
                "Args",
                (),
                {
                    "universe": universe_name,
                    "start": args.start,
                    "rebalance_frequency": None,
                    "evaluation_start": args.evaluation_start,
                    "evaluation_end": None,
                    "covariance_window": None,
                    "covariance_min_periods": None,
                },
            )(),
        )
        del backtest, evaluation

        if args.disable_quality_filter:
            prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=args.refresh_policy)
            quality_rows.append(
                {
                    "universe": universe.name,
                    "quality_filter_enabled": False,
                    "quality_reference_start": args.evaluation_start,
                    "kept_tickers": int(prices.shape[1]),
                    "excluded_tickers": 0,
                }
            )
        else:
            prices, quality_report = load_filtered_prices_for_universe(
                universe,
                evaluation_start=args.evaluation_start,
                refresh_policy=args.refresh_policy,
            )
            quality_rows.append(
                {
                    "universe": universe.name,
                    "quality_filter_enabled": bool(quality_report.enabled),
                    "quality_reference_start": quality_report.reference_start,
                    "kept_tickers": int(len(quality_report.kept_tickers)),
                    "excluded_tickers": int(len(quality_report.excluded_tickers)),
                }
            )
        matrix_date = resolve_allocation_date(prices.index, as_of_date=args.matrix_date)
        history = prices.loc[prices.index <= matrix_date]
        if history.empty:
            raise ValueError(f"No price history available on or before {matrix_date.date()} for universe '{universe.name}'.")

        loaded_num_assets = int(history.shape[1])
        estimator_window = max(2, 2 * loaded_num_assets)
        window_estimation = resolve_window_estimation_cfg(estimation, estimator_window, min_periods_mode="clamp")
        empirical_corr, sample_cov, sample_size, sample_frame = matrix_sample_bundle(
            history,
            window_estimation,
            matrix_date,
            input_type=input_mode,
            estimator_method=estimator_method,
            estimator_window=estimator_window,
        )
        num_assets = int(empirical_corr.shape[0])
        sample_vol = pd.Series(
            np.sqrt(np.clip(np.diag(sample_cov.to_numpy(dtype=float)), 0.0, None)),
            index=sample_cov.index,
            dtype=float,
        )

        method_vectors: dict[str, np.ndarray] = {}
        method_eigenvalues: dict[str, np.ndarray] = {}
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
            eigenvalues, eigenvectors = _ordered_eigenvectors(matrix, top_n=args.top_n)
            method_eigenvalues[requested_name] = eigenvalues
            method_vectors[requested_name] = eigenvectors

        for left_name, right_name in combinations(method_vectors, 2):
            left_vectors = method_vectors[left_name]
            right_vectors = method_vectors[right_name]
            left_eigenvalues = method_eigenvalues[left_name]
            right_eigenvalues = method_eigenvalues[right_name]
            limit = min(left_vectors.shape[1], right_vectors.shape[1], args.top_n)
            pair_rows: list[dict[str, object]] = []
            for rank in range(limit):
                abs_alignment, signed_alignment, max_abs_loading_diff, l2_loading_diff = _compare_vector_pair(
                    left_vectors[:, rank],
                    right_vectors[:, rank],
                )
                row = {
                    "universe": universe.name,
                    "matrix_date": matrix_date.strftime("%Y-%m-%d"),
                    "matrix_type": args.matrix_type,
                    "input_type": args.input_type,
                    "estimator_method": args.estimator_method,
                    "estimator_window": int(estimator_window),
                    "num_assets": int(num_assets),
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
                alignment_rows.append(row)
                pair_rows.append(row)

            pair_frame = pd.DataFrame(pair_rows)
            summary_rows.append(
                {
                    "universe": universe.name,
                    "matrix_date": matrix_date.strftime("%Y-%m-%d"),
                    "matrix_type": args.matrix_type,
                    "input_type": args.input_type,
                    "estimator_method": args.estimator_method,
                    "estimator_window": int(estimator_window),
                    "num_assets": int(num_assets),
                    "sample_size": int(sample_size),
                    "left_method": left_name,
                    "left_method_resolved": METHOD_ALIASES[left_name],
                    "right_method": right_name,
                    "right_method_resolved": METHOD_ALIASES[right_name],
                    "num_ranks": int(len(pair_frame)),
                    "mean_abs_alignment": float(pair_frame["abs_alignment"].mean()),
                    "min_abs_alignment": float(pair_frame["abs_alignment"].min()),
                    "max_abs_loading_diff": float(pair_frame["max_abs_loading_diff"].max()),
                    "mean_l2_loading_diff": float(pair_frame["l2_loading_diff"].mean()),
                    "num_ranks_below_0p99": int((pair_frame["abs_alignment"] < 0.99).sum()),
                    "num_ranks_below_0p95": int((pair_frame["abs_alignment"] < 0.95).sum()),
                    "num_ranks_below_0p90": int((pair_frame["abs_alignment"] < 0.90).sum()),
                }
            )

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
                    "universe": universe.name,
                    "matrix_date": matrix_date.strftime("%Y-%m-%d"),
                    "matrix_type": args.matrix_type,
                    "input_type": args.input_type,
                    "estimator_method": args.estimator_method,
                    "estimator_window": int(estimator_window),
                    "num_assets": int(num_assets),
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
                matched_alignment_rows.append(row)
                matched_pair_rows.append(row)

            matched_pair_frame = pd.DataFrame(matched_pair_rows)
            matched_summary_rows.append(
                {
                    "universe": universe.name,
                    "matrix_date": matrix_date.strftime("%Y-%m-%d"),
                    "matrix_type": args.matrix_type,
                    "input_type": args.input_type,
                    "estimator_method": args.estimator_method,
                    "estimator_window": int(estimator_window),
                    "num_assets": int(num_assets),
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
                }
            )

    alignment_frame = pd.DataFrame(alignment_rows).sort_values(
        ["universe", "left_method", "right_method", "rank"],
        ignore_index=True,
    )
    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["universe", "left_method", "right_method"],
        ignore_index=True,
    )
    matched_alignment_frame = pd.DataFrame(matched_alignment_rows).sort_values(
        ["universe", "left_method", "right_method", "left_rank"],
        ignore_index=True,
    )
    matched_summary_frame = pd.DataFrame(matched_summary_rows).sort_values(
        ["universe", "left_method", "right_method"],
        ignore_index=True,
    )
    alignment_path = output_dir / "eigenvector_alignment.csv"
    summary_path = output_dir / "eigenvector_alignment_summary.csv"
    matched_alignment_path = output_dir / "eigenvector_alignment_matched.csv"
    matched_summary_path = output_dir / "eigenvector_alignment_matched_summary.csv"
    quality_path = output_dir / "quality_filter_summary.csv"
    alignment_frame.to_csv(alignment_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    matched_alignment_frame.to_csv(matched_alignment_path, index=False)
    matched_summary_frame.to_csv(matched_summary_path, index=False)
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)

    print(f"universes: {', '.join(universes)}")
    print(f"quality_filter_enabled: {not args.disable_quality_filter}")
    print(f"evaluation_start: {args.evaluation_start or 'config/default'}")
    print(f"matrix_type: {args.matrix_type}")
    print(f"input_type: {args.input_type}")
    print(f"estimator_method: {args.estimator_method}")
    print(f"top_n: {int(args.top_n)}")
    print(f"alignment_csv: {alignment_path}")
    print(f"summary_csv: {summary_path}")
    print(f"matched_alignment_csv: {matched_alignment_path}")
    print(f"matched_summary_csv: {matched_summary_path}")
    print(f"quality_summary_csv: {quality_path}")
    print("summary_preview_rank_to_rank:")
    print(
        summary_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("summary_preview_matched:")
    print(
        matched_summary_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
