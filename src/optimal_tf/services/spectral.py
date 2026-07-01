from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.scripts.common import (
    eigenvalue_rows,
    matrix_sample,
    merge_common_overrides,
    parse_windows,
    render_scree_overview,
    resolve_target_dates,
    resolve_window_estimation_cfg,
    validate_methods,
)
from trading_core.risk import clean_correlation_matrix
from trading_core.reporting.plots import plt

from .io import ensure_output_dir, write_request_json
from .models import (
    RunArtifacts,
    SpectrumByCleanerRequest,
    SpectrumByCleanerResult,
    SpectrumByWindowRequest,
    SpectrumByWindowResult,
)


def _resolve_config_overrides(request, *, default_universe, default_estimation, default_backtest, default_evaluation):
    return merge_common_overrides(
        default_universe,
        default_estimation,
        default_backtest,
        default_evaluation,
        type(
            "Args",
            (),
            {
                "universe": request.universe,
                "start": request.start,
                "rebalance_frequency": request.rebalance_frequency,
                "evaluation_start": request.evaluation_start,
                "evaluation_end": request.evaluation_end,
            },
        )(),
    )


def _eigenvalue_frame(
    empirical_corr: pd.DataFrame,
    sample_size: int,
    sample_frame: pd.DataFrame,
    estimation,
    methods: list[str],
    matrix_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        cleaned = clean_correlation_matrix(
            empirical_corr,
            data=sample_frame,
            sample_size=sample_size,
            method=method,
            linear_shrinkage=estimation.linear_shrinkage,
            bandwidth=estimation.rie_bandwidth,
        )
        eigenvalues = np.linalg.eigvalsh(cleaned.to_numpy(dtype=float))[::-1]
        total = float(np.sum(eigenvalues))
        cumulative = np.cumsum(eigenvalues)
        for rank, eigenvalue in enumerate(eigenvalues, start=1):
            rows.append(
                {
                    "date": matrix_date.strftime("%Y-%m-%d"),
                    "method": method,
                    "rank": rank,
                    "eigenvalue": float(eigenvalue),
                    "variance_share": float(eigenvalue / total) if total else 0.0,
                    "cumulative_variance_share": float(cumulative[rank - 1] / total) if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _render_cleaner_scree_plot(frame: pd.DataFrame, output_path: Path, *, matrix_date: pd.Timestamp, log_scale: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot = frame.pivot(index="rank", columns="method", values="eigenvalue")
    for method in pivot.columns:
        ax.plot(pivot.index, pivot[method], label=str(method), linewidth=2.0)
    ax.set_title(f"Cleaner spectrum scree plot ({matrix_date.date()})")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def run_spectrum_by_cleaner(request: SpectrumByCleanerRequest) -> SpectrumByCleanerResult:
    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(request.config_path)
    del allocation, compare, output
    universe, estimation, backtest, evaluation = _resolve_config_overrides(
        request,
        default_universe=universe,
        default_estimation=estimation,
        default_backtest=backtest,
        default_evaluation=evaluation,
    )
    if request.linear_shrinkage is not None:
        estimation = replace(estimation, linear_shrinkage=float(request.linear_shrinkage))
    methods = request.methods or [estimation.cleaning_method]
    validate_methods(methods)

    prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=request.refresh_policy)
    target_dates = resolve_target_dates(prices, evaluation)
    if len(target_dates) == 0:
        raise ValueError("No evaluation rebalance dates available for the scree plot window.")
    matrix_date = pd.Timestamp(request.matrix_date) if request.matrix_date is not None else target_dates[-1]
    empirical_corr, sample_size, sample_frame = matrix_sample(prices, estimation, matrix_date)
    frame = _eigenvalue_frame(empirical_corr, sample_size, sample_frame, estimation, methods, matrix_date)

    outdir = ensure_output_dir(request.output_dir or "output/optimal_tf/spectral/by_cleaner")
    files: dict[str, Path] = {}
    if outdir is not None:
        csv_path = outdir / "eigenvalue_scree.csv"
        plot_path = outdir / "eigenvalue_scree.png"
        frame.to_csv(csv_path, index=False)
        _render_cleaner_scree_plot(frame, plot_path, matrix_date=matrix_date, log_scale=request.log_scale)
        summary = {
            "universe": universe.name,
            "matrix_date": matrix_date.strftime("%Y-%m-%d"),
            "methods": methods,
            "num_assets": int(empirical_corr.shape[0]),
            "sample_size": int(sample_size),
            "log_scale": bool(request.log_scale),
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        req = write_request_json(outdir, request)
        files = {
            "csv": csv_path,
            "plot": plot_path,
            "summary": outdir / "summary.json",
        }
        if req is not None:
            files["request"] = req

    return SpectrumByCleanerResult(
        request=request,
        universe=universe.name,
        matrix_date=matrix_date,
        methods=methods,
        eigenvalue_frame=frame,
        num_assets=int(empirical_corr.shape[0]),
        sample_size=int(sample_size),
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )


def run_spectrum_by_window(request: SpectrumByWindowRequest) -> SpectrumByWindowResult:
    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(request.config_path)
    del allocation, compare, output
    universe, estimation, backtest, evaluation = _resolve_config_overrides(
        request,
        default_universe=universe,
        default_estimation=estimation,
        default_backtest=backtest,
        default_evaluation=evaluation,
    )
    if request.linear_shrinkage is not None:
        estimation = replace(estimation, linear_shrinkage=float(request.linear_shrinkage))
    method = request.method or estimation.cleaning_method
    validate_methods([method])
    windows = request.windows or [estimation.covariance_window or estimation.corr_span]
    windows = parse_windows(",".join(str(item) for item in windows))

    prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=request.refresh_policy)
    target_dates = resolve_target_dates(prices, evaluation)
    if len(target_dates) == 0:
        raise ValueError("No evaluation rebalance dates available for the spectral comparison window.")
    matrix_date = pd.Timestamp(request.matrix_date) if request.matrix_date is not None else target_dates[-1]

    rows: list[dict[str, float | int | str]] = []
    for window in windows:
        window_estimation = resolve_window_estimation_cfg(estimation, window, min_periods_mode=request.min_periods_mode)
        empirical_corr, sample_size, sample_frame = matrix_sample(prices, window_estimation, matrix_date)
        rows.extend(
            eigenvalue_rows(
                empirical_corr,
                sample_size,
                sample_frame,
                window_estimation,
                [method],
                matrix_date=matrix_date,
            )
        )
    frame = pd.DataFrame(rows).sort_values(["covariance_window", "method", "rank"]).reset_index(drop=True)

    outdir = ensure_output_dir(request.output_dir or "output/optimal_tf/spectral/by_window")
    files: dict[str, Path] = {}
    if outdir is not None:
        csv_path = outdir / "window_scree.csv"
        plot_path = render_scree_overview(frame, outdir / "window_scree_overview.png", log_scale=request.log_scale)
        frame.to_csv(csv_path, index=False)
        summary = {
            "universe": universe.name,
            "matrix_date": matrix_date.strftime("%Y-%m-%d"),
            "method": method,
            "windows": windows,
            "log_scale": bool(request.log_scale),
            "min_periods_mode": request.min_periods_mode,
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        req = write_request_json(outdir, request)
        files = {
            "csv": csv_path,
            "plot": plot_path,
            "summary": outdir / "summary.json",
        }
        if req is not None:
            files["request"] = req

    return SpectrumByWindowResult(
        request=request,
        universe=universe.name,
        matrix_date=matrix_date,
        method=method,
        windows=windows,
        scree_frame=frame,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )
