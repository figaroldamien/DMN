from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.scripts.common import resolve_window_estimation_cfg
from optimal_tf.scripts.common.benchmark import matrix_sample_bundle
from optimal_tf.strategies.common import resolve_allocation_date
from trading_core.features import compute_returns
from trading_core.risk import clean_correlation_matrix

from .compare_cleaner_eigenvectors import _optimal_rank_matching, _ordered_eigenvectors

DEFAULT_OUTPUT_DIR = "output/optimal_tf/inspection/sp500_quality_filter"
DEFAULT_TOP_NS = (5, 10)
DEFAULT_THRESHOLDS = (1.0, 0.95, 0.90)


def _parse_float_csv(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one float value is required.")
    return values


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _audit_recent_window(prices: pd.DataFrame, *, window_days: int) -> pd.DataFrame:
    recent = prices.tail(window_days)
    returns = compute_returns(prices)
    rows: list[dict[str, object]] = []
    last_date = prices.index.max()
    for ticker in prices.columns:
        series = prices[ticker]
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()
        recent_series = recent[ticker]
        recent_coverage = float(recent_series.notna().mean()) if len(recent_series) else np.nan
        internal_missing_recent = 0
        recent_first = recent_series.first_valid_index()
        recent_last = recent_series.last_valid_index()
        if recent_first is not None and recent_last is not None:
            internal_missing_recent = int(recent_series.loc[recent_first:recent_last].isna().sum())
        abs_returns = returns[ticker].abs()
        rows.append(
            {
                "ticker": ticker,
                "first_valid": first_valid,
                "last_valid": last_valid,
                "has_latest_price": bool(pd.notna(series.loc[last_date])),
                "recent_coverage_ratio": recent_coverage,
                "recent_valid_days": int(recent_series.notna().sum()),
                "recent_internal_missing": int(internal_missing_recent),
                "max_abs_return": float(abs_returns.max()) if abs_returns.notna().any() else np.nan,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["recent_coverage_ratio", "ticker"], ascending=[True, True], ignore_index=True)
    return frame


def _matched_alignment_summary(
    prices: pd.DataFrame,
    *,
    matrix_date: pd.Timestamp,
    estimation,
    top_n: int,
) -> dict[str, object]:
    estimator_window = max(2, 2 * prices.shape[1])
    window_estimation = resolve_window_estimation_cfg(estimation, estimator_window, min_periods_mode="clamp")
    empirical_corr, sample_cov, sample_size, sample_frame = matrix_sample_bundle(
        prices,
        window_estimation,
        matrix_date,
        input_type="normalized",
        estimator_method="window_sample",
        estimator_window=estimator_window,
    )
    empirical_corr = clean_correlation_matrix(
        empirical_corr,
        data=sample_frame,
        sample_size=sample_size,
        method="empirical",
        linear_shrinkage=0.0,
        bandwidth=window_estimation.rie_bandwidth,
    )
    rie_corr = clean_correlation_matrix(
        empirical_corr,
        data=sample_frame,
        sample_size=sample_size,
        method="rie_reference",
        linear_shrinkage=window_estimation.linear_shrinkage,
        bandwidth=window_estimation.rie_bandwidth,
    )
    empirical_vals, empirical_vecs = _ordered_eigenvectors(empirical_corr, top_n=top_n)
    rie_vals, rie_vecs = _ordered_eigenvectors(rie_corr, top_n=top_n)
    pairs = _optimal_rank_matching(empirical_vecs, rie_vecs)
    alignments: list[float] = []
    mapping: list[str] = []
    for left_rank, right_rank, _ in pairs:
        left = empirical_vecs[:, left_rank]
        right = rie_vecs[:, right_rank]
        alignment = float(abs(np.dot(left, right)))
        alignments.append(alignment)
        mapping.append(f"{left_rank + 1}->{right_rank + 1}")
    return {
        "top_n": int(top_n),
        "num_assets": int(empirical_corr.shape[0]),
        "sample_size": int(sample_size),
        "estimator_window": int(estimator_window),
        "mean_abs_alignment": float(np.mean(alignments)),
        "min_abs_alignment": float(np.min(alignments)),
        "num_ranks_below_0p99": int(sum(value < 0.99 for value in alignments)),
        "num_ranks_below_0p95": int(sum(value < 0.95 for value in alignments)),
        "num_ranks_below_0p90": int(sum(value < 0.90 for value in alignments)),
        "rank_mapping": ",".join(mapping),
        "leading_empirical_eigenvalue": float(empirical_vals[0]) if len(empirical_vals) else np.nan,
        "leading_rie_eigenvalue": float(rie_vals[0]) if len(rie_vals) else np.nan,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit SP500 price quality and compare cleaner alignment before/after filtering.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--matrix-date", type=str, default=None)
    parser.add_argument("--window-days", type=int, default=1000)
    parser.add_argument("--coverage-thresholds", type=str, default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--top-ns", type=str, default=",".join(str(value) for value in DEFAULT_TOP_NS))
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-policy", choices=("auto", "always", "never"), default="auto")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    top_ns = _parse_int_csv(args.top_ns)
    coverage_thresholds = _parse_float_csv(args.coverage_thresholds)

    universe, estimation, *_ = load_config(args.config)
    universe = replace(universe, name="sp500", start=args.start or universe.start)
    prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=args.refresh_policy)
    matrix_date = resolve_allocation_date(prices.index, as_of_date=args.matrix_date)
    history = prices.loc[prices.index <= matrix_date].copy()
    if history.empty:
        raise ValueError(f"No price history available on or before {matrix_date.date()} for sp500.")

    audit_frame = _audit_recent_window(history, window_days=int(args.window_days))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "sp500_quality_audit.csv"
    audit_frame.to_csv(audit_path, index=False)

    rows: list[dict[str, object]] = []
    baseline_columns = [column for column in history.columns if pd.notna(history[column].ffill().iloc[-1])]
    baseline_prices = history.loc[:, baseline_columns].copy()
    for top_n in top_ns:
        payload = _matched_alignment_summary(
            baseline_prices,
            matrix_date=matrix_date,
            estimation=estimation,
            top_n=int(top_n),
        )
        rows.append(
            {
                "scenario": "baseline_latest_available",
                "coverage_threshold": np.nan,
                **payload,
            }
        )

    for threshold in coverage_thresholds:
        eligible = audit_frame[
            (audit_frame["has_latest_price"])
            & (audit_frame["recent_coverage_ratio"] >= float(threshold))
            & (audit_frame["recent_internal_missing"] == 0)
        ]["ticker"].tolist()
        filtered_prices = history.loc[:, eligible].copy()
        for top_n in top_ns:
            payload = _matched_alignment_summary(
                filtered_prices,
                matrix_date=matrix_date,
                estimation=estimation,
                top_n=int(top_n),
            )
            rows.append(
                {
                    "scenario": "filtered_recent_coverage",
                    "coverage_threshold": float(threshold),
                    **payload,
                }
            )

    summary_frame = pd.DataFrame(rows).sort_values(["top_n", "scenario", "coverage_threshold"], ignore_index=True)
    summary_path = output_dir / "sp500_quality_filter_summary.csv"
    summary_frame.to_csv(summary_path, index=False)

    print(f"matrix_date: {matrix_date.date()}")
    print(f"window_days: {int(args.window_days)}")
    print(f"audit_csv: {audit_path}")
    print(f"summary_csv: {summary_path}")
    print("summary:")
    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
