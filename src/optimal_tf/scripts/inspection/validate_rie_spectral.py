from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from optimal_tf.config_io import load_config
from optimal_tf.data_quality import load_filtered_prices_for_universe
from optimal_tf.estimators.rie_spectral import clean_correlation_matrix_rie_spectral
from optimal_tf.scripts.common import merge_common_overrides, resolve_window_estimation_cfg
from optimal_tf.scripts.common.benchmark import matrix_sample_bundle
from optimal_tf.services.inspection import UNIVERSE_COMPONENTS
from optimal_tf.strategies.common import resolve_allocation_date
from trading_core.risk import clean_correlation_matrix

from .compare_cleaner_eigenvectors import (
    ESTIMATOR_METHOD_ALIASES,
    INPUT_TYPE_ALIASES,
    _compare_vector_pair,
    _optimal_rank_matching,
)

DEFAULT_UNIVERSES = tuple(sorted(name for name in UNIVERSE_COMPONENTS if name != "table8_all"))
DEFAULT_OUTPUT_DIR = "output/optimal_tf/inspection/rie_spectral_validation"
DEFAULT_TOP_N = 10


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_universes(raw_universes: list[str]) -> list[str]:
    universes = raw_universes or list(DEFAULT_UNIVERSES)
    invalid = [name for name in universes if name not in UNIVERSE_COMPONENTS]
    if invalid:
        raise ValueError(f"Unknown universes {invalid}. Allowed values: {sorted(UNIVERSE_COMPONENTS)}")
    return universes


def _leading_vectors_from_frame(matrix: pd.DataFrame, *, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.to_numpy(dtype=float))
    order = np.argsort(eigenvalues)[::-1]
    limit = min(max(1, int(top_n)), eigenvectors.shape[1])
    return eigenvalues[order][:limit].astype(float), eigenvectors[:, order][:, :limit].astype(float)


def _leading_vectors_from_result(eigenvalues: np.ndarray, eigenvectors: pd.DataFrame, *, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    limit = min(max(1, int(top_n)), eigenvectors.shape[1], len(eigenvalues))
    return eigenvalues[:limit].astype(float), eigenvectors.to_numpy(dtype=float)[:, :limit].astype(float)


def _same_rank_summary(left_values: np.ndarray, left_vectors: np.ndarray, right_values: np.ndarray, right_vectors: np.ndarray) -> dict[str, object]:
    limit = min(left_vectors.shape[1], right_vectors.shape[1], len(left_values), len(right_values))
    rows: list[dict[str, float | int]] = []
    for rank in range(limit):
        abs_alignment, signed_alignment, max_abs_loading_diff, l2_loading_diff = _compare_vector_pair(
            left_vectors[:, rank],
            right_vectors[:, rank],
        )
        rows.append(
            {
                "rank": int(rank + 1),
                "left_eigenvalue": float(left_values[rank]),
                "right_eigenvalue": float(right_values[rank]),
                "abs_alignment": abs_alignment,
                "signed_alignment": signed_alignment,
                "max_abs_loading_diff": max_abs_loading_diff,
                "l2_loading_diff": l2_loading_diff,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"num_ranks": 0}
    return {
        "num_ranks": int(len(frame)),
        "mean_abs_alignment": float(frame["abs_alignment"].mean()),
        "min_abs_alignment": float(frame["abs_alignment"].min()),
        "mean_signed_alignment": float(frame["signed_alignment"].mean()),
        "mean_max_abs_loading_diff": float(frame["max_abs_loading_diff"].mean()),
        "mean_l2_loading_diff": float(frame["l2_loading_diff"].mean()),
    }


def _matched_summary(left_values: np.ndarray, left_vectors: np.ndarray, right_values: np.ndarray, right_vectors: np.ndarray) -> dict[str, object]:
    limit = min(left_vectors.shape[1], right_vectors.shape[1], len(left_values), len(right_values))
    rows: list[dict[str, float | int]] = []
    for left_rank, right_rank, matched_abs_alignment in _optimal_rank_matching(left_vectors[:, :limit], right_vectors[:, :limit]):
        abs_alignment, signed_alignment, max_abs_loading_diff, l2_loading_diff = _compare_vector_pair(
            left_vectors[:, left_rank],
            right_vectors[:, right_rank],
        )
        rows.append(
            {
                "left_rank": int(left_rank + 1),
                "right_rank": int(right_rank + 1),
                "left_eigenvalue": float(left_values[left_rank]),
                "right_eigenvalue": float(right_values[right_rank]),
                "abs_alignment": abs_alignment,
                "signed_alignment": signed_alignment,
                "matched_abs_alignment": matched_abs_alignment,
                "max_abs_loading_diff": max_abs_loading_diff,
                "l2_loading_diff": l2_loading_diff,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"num_ranks": 0}
    return {
        "num_ranks": int(len(frame)),
        "mean_abs_alignment": float(frame["abs_alignment"].mean()),
        "min_abs_alignment": float(frame["abs_alignment"].min()),
        "mean_signed_alignment": float(frame["signed_alignment"].mean()),
        "mean_matched_abs_alignment": float(frame["matched_abs_alignment"].mean()),
        "same_rank_count": int((frame["left_rank"] == frame["right_rank"]).sum()),
        "rank_mapping": ",".join(f"{int(row.left_rank)}->{int(row.right_rank)}" for row in frame.itertuples(index=False)),
    }


def _matrix_diff_summary(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    diff = left.to_numpy(dtype=float) - right.to_numpy(dtype=float)
    return {
        "fro_diff": float(np.linalg.norm(diff, ord="fro")),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the rie_spectral implementation on filtered universes.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universes", type=str, default=",".join(DEFAULT_UNIVERSES))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--input-type", type=str, default="normalized_returns")
    parser.add_argument("--estimator-method", choices=("sample_window", "ewma"), default="sample_window")
    parser.add_argument("--refresh-policy", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    universes = _resolve_universes(_parse_csv(args.universes))
    input_mode = INPUT_TYPE_ALIASES.get(args.input_type.strip().lower())
    if input_mode is None:
        raise ValueError(f"Unknown input type '{args.input_type}'.")
    estimator_method = ESTIMATOR_METHOD_ALIASES.get(args.estimator_method.strip().lower())
    if estimator_method is None:
        raise ValueError(f"Unknown estimator method '{args.estimator_method}'.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
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
        del backtest

        prices, quality_report = load_filtered_prices_for_universe(
            universe,
            evaluation_start=evaluation.evaluation_start,
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

        num_assets = int(history.shape[1])
        estimator_window = max(2, 2 * num_assets)
        window_estimation = resolve_window_estimation_cfg(estimation, estimator_window, min_periods_mode="clamp")
        empirical_corr, sample_cov, sample_size, sample_frame = matrix_sample_bundle(
            history,
            window_estimation,
            matrix_date,
            input_type=input_mode,
            estimator_method=estimator_method,
            estimator_window=estimator_window,
        )
        del sample_cov

        empirical_vals, empirical_vecs = _leading_vectors_from_frame(empirical_corr, top_n=args.top_n)
        linear_corr = clean_correlation_matrix(
            empirical_corr,
            data=sample_frame,
            sample_size=sample_size,
            method="linear_shrinkage",
            linear_shrinkage=estimation.linear_shrinkage,
            bandwidth=estimation.rie_bandwidth,
        )
        linear_vals, linear_vecs = _leading_vectors_from_frame(linear_corr, top_n=args.top_n)

        rie_reference_corr = clean_correlation_matrix(
            empirical_corr,
            data=sample_frame,
            sample_size=sample_size,
            method="rie_reference",
            linear_shrinkage=estimation.linear_shrinkage,
            bandwidth=estimation.rie_bandwidth,
        )
        rie_reference_vals, rie_reference_vecs = _leading_vectors_from_frame(rie_reference_corr, top_n=args.top_n)

        rie_spectral = clean_correlation_matrix_rie_spectral(empirical_corr, sample_size=sample_size)
        rie_spectral_vals, rie_spectral_vecs = _leading_vectors_from_result(
            rie_spectral.cleaned.eigenvalues,
            rie_spectral.cleaned.eigenvectors,
            top_n=args.top_n,
        )
        if rie_spectral.post_projection is None:
            raise ValueError("rie_spectral validation expects a post_projection decomposition.")
        rie_post_vals, rie_post_vecs = _leading_vectors_from_result(
            rie_spectral.post_projection.eigenvalues,
            rie_spectral.post_projection.eigenvectors,
            top_n=args.top_n,
        )

        matrix_diff = _matrix_diff_summary(rie_spectral.cleaned_matrix, rie_reference_corr)
        summary_rows.append(
            {
                "universe": universe.name,
                "matrix_date": matrix_date.strftime("%Y-%m-%d"),
                "input_type": args.input_type,
                "estimator_method": args.estimator_method,
                "estimator_window": int(estimator_window),
                "num_assets": int(empirical_corr.shape[0]),
                "sample_size": int(sample_size),
                "top_n": int(args.top_n),
                "rie_spectral_postprocess_steps": ",".join(rie_spectral.postprocess_steps),
                **matrix_diff,
                "empirical_vs_linear_same_rank_mean_abs_alignment": _same_rank_summary(
                    empirical_vals,
                    empirical_vecs,
                    linear_vals,
                    linear_vecs,
                ).get("mean_abs_alignment"),
                "empirical_vs_rie_spectral_same_rank_mean_abs_alignment": _same_rank_summary(
                    empirical_vals,
                    empirical_vecs,
                    rie_spectral_vals,
                    rie_spectral_vecs,
                ).get("mean_abs_alignment"),
                "empirical_vs_rie_spectral_matched_mean_abs_alignment": _matched_summary(
                    empirical_vals,
                    empirical_vecs,
                    rie_spectral_vals,
                    rie_spectral_vecs,
                ).get("mean_abs_alignment"),
                "empirical_vs_rie_spectral_same_rank_count": _matched_summary(
                    empirical_vals,
                    empirical_vecs,
                    rie_spectral_vals,
                    rie_spectral_vecs,
                ).get("same_rank_count"),
                "empirical_vs_rie_spectral_rank_mapping": _matched_summary(
                    empirical_vals,
                    empirical_vecs,
                    rie_spectral_vals,
                    rie_spectral_vecs,
                ).get("rank_mapping"),
                "linear_vs_rie_spectral_matched_mean_abs_alignment": _matched_summary(
                    linear_vals,
                    linear_vecs,
                    rie_spectral_vals,
                    rie_spectral_vecs,
                ).get("mean_abs_alignment"),
                "rie_reference_vs_rie_post_same_rank_mean_abs_alignment": _same_rank_summary(
                    rie_reference_vals,
                    rie_reference_vecs,
                    rie_post_vals,
                    rie_post_vecs,
                ).get("mean_abs_alignment"),
                "rie_reference_vs_rie_post_matched_mean_abs_alignment": _matched_summary(
                    rie_reference_vals,
                    rie_reference_vecs,
                    rie_post_vals,
                    rie_post_vecs,
                ).get("mean_abs_alignment"),
                "rie_reference_vs_rie_post_same_rank_count": _matched_summary(
                    rie_reference_vals,
                    rie_reference_vecs,
                    rie_post_vals,
                    rie_post_vecs,
                ).get("same_rank_count"),
                "rie_reference_vs_rie_post_rank_mapping": _matched_summary(
                    rie_reference_vals,
                    rie_reference_vecs,
                    rie_post_vals,
                    rie_post_vecs,
                ).get("rank_mapping"),
                "rie_reference_vs_rie_post_eigenvalue_max_abs_diff": float(
                    np.max(np.abs(rie_reference_vals[: len(rie_post_vals)] - rie_post_vals[: len(rie_reference_vals)]))
                ),
                "rie_spectral_cleaned_vs_post_eigenvalue_max_abs_diff": float(
                    np.max(np.abs(rie_spectral_vals[: len(rie_post_vals)] - rie_post_vals[: len(rie_spectral_vals)]))
                ),
            }
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values("universe")
    quality_frame = pd.DataFrame(quality_rows).sort_values("universe")

    summary_path = output_dir / "rie_spectral_validation_summary.csv"
    quality_path = output_dir / "quality_filter_summary.csv"
    json_path = output_dir / "rie_spectral_validation_summary.json"
    summary_frame.to_csv(summary_path, index=False)
    quality_frame.to_csv(quality_path, index=False)
    json_path.write_text(summary_frame.to_json(orient="records", indent=2), encoding="utf-8")

    print(f"summary_csv: {summary_path}")
    print(f"quality_csv: {quality_path}")
    print(f"summary_json: {json_path}")
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
