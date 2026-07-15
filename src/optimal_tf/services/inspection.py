from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_tickers_data.components import (
    CAC40_COMPONENTS,
    DATASET_COMPONENTS,
    DJI_COMPONENTS,
    EUROSTOXX50_COMPONENTS,
    EUROSTOXX600_COMPONENTS,
    FUTURES_COMPONENTS,
    INDEX_COMPONENTS,
    NASDAQ100_COMPONENTS,
    SBF120_COMPONENTS,
    SP500_COMPONENTS,
    WORLD_INDEX_COMPONENTS,
)
from optimal_tf.config_io import load_config
from optimal_tf.data_quality import load_filtered_prices_for_universe
from optimal_tf.estimators import clean_correlation_matrix_rich
from optimal_tf.features import trend_ema_signal
from optimal_tf.scripts.common import (
    matrix_sample_bundle,
    matrix_sample,
    merge_common_overrides,
    parse_windows,
    resolve_target_dates,
    resolve_window_estimation_cfg,
)
from trading_core.risk import (
    clean_correlation_matrix,
    correlation_to_covariance,
    marchenko_pastur_law,
    supported_cleaning_methods,
)
from trading_core.reporting import cumulative_nav, evaluation_metrics
from trading_core.reporting.plots import plt

from optimal_tf.strategies.common import resolve_allocation_date, sanitized_normalized_returns

from .io import ensure_output_dir, write_json, write_quality_artifacts, write_request_json
from .models import (
    EigenvectorInspectionRequest,
    EigenvectorInspectionResult,
    InspectionIntervalRequest,
    InspectionIntervalResult,
    InspectionSnapshotRequest,
    InspectionSnapshotResult,
    RunArtifacts,
)

DEFAULT_WINDOWS = (40, 60, 80, 120, 252, 504, 1200)
MATRIX_INPUT_OPTIONS = {"normalized_returns", "raw_returns"}
MATRIX_INPUT_ALIASES = {
    "normalized": "normalized_returns",
    "normalized_returns": "normalized_returns",
    "raw": "raw_returns",
    "raw_returns": "raw_returns",
}
MATRIX_TYPE_OPTIONS = {"correlation", "covariance"}
ESTIMATOR_METHOD_OPTIONS = {"sample_window", "ewma"}
ESTIMATOR_METHOD_ALIASES = {
    "window_sample": "sample_window",
    "sample_window": "sample_window",
    "ewma_cross": "ewma",
    "ewma": "ewma",
}
INSPECTION_CLEANING_OPTIONS = {"empirical", "linear_shrinkage", "rie_spectral", "rie_reference"}
INSPECTION_CLEANING_ALIASES = {
    "empirical": "empirical",
    "linear_shrinkage": "linear_shrinkage",
    "rie": "rie_spectral",
    "rie_spectral": "rie_spectral",
    "rie_reference": "rie_reference",
}
UNIVERSE_COMPONENTS = {
    "nasdaq100": NASDAQ100_COMPONENTS,
    "cac40": CAC40_COMPONENTS,
    "dji": DJI_COMPONENTS,
    "eurostoxx50": EUROSTOXX50_COMPONENTS,
    "eurostoxx600": EUROSTOXX600_COMPONENTS,
    "sbf120": SBF120_COMPONENTS,
    "sp500": SP500_COMPONENTS,
    "index": INDEX_COMPONENTS,
    "futures": FUTURES_COMPONENTS,
    "world_index": WORLD_INDEX_COMPONENTS,
    "dataset_all": DATASET_COMPONENTS,
    "table8_all": DATASET_COMPONENTS,
}


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
                "universe": getattr(request, "universe", None),
                "start": getattr(request, "start", None),
                "rebalance_frequency": getattr(request, "rebalance_frequency", None),
                "evaluation_start": getattr(request, "evaluation_start", None),
                "evaluation_end": getattr(request, "evaluation_end", None),
            },
        )(),
    )


def _resolve_inspection_cleaning_method(value: str | None, *, fallback: str) -> str:
    raw = str(value or fallback).strip().lower()
    resolved = INSPECTION_CLEANING_ALIASES.get(raw)
    if resolved is None or resolved not in INSPECTION_CLEANING_OPTIONS:
        raise ValueError(f"Unknown cleaning_method {raw!r}.")
    return resolved


def _resolve_input_type(value: str | None, *, fallback: str = "normalized_returns") -> str:
    raw = str(value or fallback).strip().lower()
    resolved = MATRIX_INPUT_ALIASES.get(raw)
    if resolved is None or resolved not in MATRIX_INPUT_OPTIONS:
        raise ValueError(f"Unknown input_type {raw!r}.")
    return resolved


def _resolve_matrix_type(value: str | None, *, fallback: str = "correlation") -> str:
    resolved = str(value or fallback).strip().lower()
    if resolved not in MATRIX_TYPE_OPTIONS:
        raise ValueError(f"Unknown matrix_type {resolved!r}.")
    return resolved


def _resolve_estimator_method(value: str | None, *, fallback: str = "sample_window") -> str:
    raw = str(value or fallback).strip().lower()
    resolved = ESTIMATOR_METHOD_ALIASES.get(raw)
    if resolved is None or resolved not in ESTIMATOR_METHOD_OPTIONS:
        raise ValueError(f"Unknown estimator_method {raw!r}.")
    return resolved


def _resolve_estimator_window(request, estimation) -> int:
    raw = getattr(request, "estimator_window", None)
    if raw is None:
        raw = getattr(request, "covariance_window", None)
    if raw is None:
        raw = estimation.covariance_window or 252
    window = int(raw)
    if window <= 0:
        raise ValueError("estimator_window must be strictly positive.")
    return window


def _resolve_linear_shrinkage_intensity(request, estimation) -> float:
    raw = getattr(request, "linear_shrinkage_intensity", None)
    if raw is None:
        raw = getattr(request, "linear_shrinkage", None)
    if raw is None:
        raw = estimation.linear_shrinkage
    return float(raw)


def _matrix_sample_input_mode(input_type: str) -> str:
    return "normalized" if input_type == "normalized_returns" else "raw"


def _matrix_sample_estimator_mode(estimator_method: str) -> str:
    return "window_sample" if estimator_method == "sample_window" else "ewma_cross"


def _sector_metadata(universe: str, tickers: list[str]) -> pd.DataFrame:
    components = UNIVERSE_COMPONENTS.get(universe, {})
    rows: list[dict[str, str]] = []
    for ticker in tickers:
        meta = components.get(ticker, {})
        sector = str(meta.get("sector", "")).strip()
        sub_sector = str(meta.get("sub_sector", "")).strip()
        category = str(meta.get("category", "")).strip()
        sub_category = str(meta.get("sub_category", "")).strip()
        rows.append(
            {
                "ticker": ticker,
                "sector": sector or category or "zzz_unknown",
                "sub_sector": sub_sector or sub_category or "zzz_unknown",
                "category": category or "zzz_unknown",
                "sub_category": sub_category or "zzz_unknown",
            }
        )
    frame = pd.DataFrame(rows).set_index("ticker")
    return frame.sort_values(["sector", "sub_sector"])


def _selection_suffix(selection_mode: str, *, cumulative_variance_pct: float, top_n: int) -> str:
    if selection_mode == "mp":
        return "mp_outlier"
    if selection_mode == "cumulative_variance":
        return f"cumvar_{cumulative_variance_pct:g}".replace(".", "p")
    if selection_mode == "top_n":
        return f"top_{top_n}"
    raise ValueError(f"Unknown selection mode {selection_mode!r}")


def _absolute_vector_alignment(left: pd.Series, right: pd.Series) -> float | None:
    common = left.index.intersection(right.index)
    if common.empty:
        return None
    left_values = left.loc[common].to_numpy(dtype=float)
    right_values = right.loc[common].to_numpy(dtype=float)
    left_norm = float(np.linalg.norm(left_values))
    right_norm = float(np.linalg.norm(right_values))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return float(abs(np.dot(left_values / left_norm, right_values / right_norm)))


def _variogram_frame(series_frame: pd.DataFrame, *, max_lag: int) -> pd.DataFrame:
    if series_frame.empty:
        return pd.DataFrame(columns=["rank", "lag", "semivariance", "num_pairs"])
    rows: list[dict[str, float | int]] = []
    effective_max_lag = max(0, min(int(max_lag), len(series_frame) - 1))
    ordered = series_frame.sort_index()
    for column in ordered.columns:
        rank_label = str(column)
        rank = int(rank_label.removeprefix("rank_")) if rank_label.startswith("rank_") else rank_label
        values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
        for lag in range(effective_max_lag + 1):
            if lag == 0:
                rows.append({"rank": rank, "lag": 0, "semivariance": 0.0, "num_pairs": int(len(values))})
                continue
            diffs = values[lag:] - values[:-lag]
            valid = np.isfinite(diffs)
            num_pairs = int(np.sum(valid))
            semivariance = float(0.5 * np.mean(np.square(diffs[valid]))) if num_pairs else np.nan
            rows.append(
                {
                    "rank": rank,
                    "lag": lag,
                    "semivariance": semivariance,
                    "num_pairs": num_pairs,
                }
            )
    return pd.DataFrame(rows)


def _select_eigen_positions(
    eigenvalues: np.ndarray,
    mp,
    *,
    selection_mode: str,
    cumulative_variance_pct: float,
    top_n: int,
) -> list[int]:
    if len(eigenvalues) == 0:
        return []
    if selection_mode == "mp":
        return [idx for idx, value in enumerate(eigenvalues) if value > mp.lambda_plus]
    if selection_mode == "cumulative_variance":
        threshold = float(np.clip(cumulative_variance_pct, 0.0, 100.0)) / 100.0
        total = float(np.sum(eigenvalues))
        if total <= 0.0:
            return []
        cumulative = np.cumsum(eigenvalues) / total
        cutoff = int(np.searchsorted(cumulative, threshold, side="left")) + 1
        cutoff = max(1, min(cutoff, len(eigenvalues)))
        return list(range(cutoff))
    if selection_mode == "top_n":
        cutoff = max(0, min(int(top_n), len(eigenvalues)))
        return list(range(cutoff))
    raise ValueError(f"Unknown selection mode {selection_mode!r}")


def _cleaned_eigendecomposition(empirical_corr: pd.DataFrame, sample_size: int, sample_frame: pd.DataFrame, estimation, method: str):
    cleaned = clean_correlation_matrix(
        empirical_corr,
        data=sample_frame,
        sample_size=sample_size,
        method=method,
        linear_shrinkage=estimation.linear_shrinkage,
        bandwidth=estimation.rie_bandwidth,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(cleaned.to_numpy(dtype=float))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    mp = marchenko_pastur_law(len(cleaned), sample_size, variance=1.0)
    return cleaned, eigenvalues, eigenvectors, mp


def _spectrum_rows(empirical_corr, sample_size, sample_frame, estimation, method: str, *, matrix_date: pd.Timestamp):
    cleaned, eigenvalues, _, mp = _cleaned_eigendecomposition(empirical_corr, sample_size, sample_frame, estimation, method)
    bulk_rank = int(np.sum(eigenvalues > mp.lambda_plus))
    total = float(np.sum(eigenvalues))
    cumulative = np.cumsum(eigenvalues)
    rows: list[dict[str, float | int | str]] = []
    for rank, eigenvalue in enumerate(eigenvalues, start=1):
        rows.append(
            {
                "date": matrix_date.strftime("%Y-%m-%d"),
                "method": method,
                "covariance_window": int(estimation.covariance_window or 0),
                "covariance_min_periods": int(estimation.covariance_min_periods),
                "num_assets": int(len(cleaned)),
                "sample_size": int(sample_size),
                "rank": rank,
                "eigenvalue": float(eigenvalue),
                "variance_share": float(eigenvalue / total) if total else 0.0,
                "cumulative_variance_share": float(cumulative[rank - 1] / total) if total else 0.0,
                "mp_lambda_minus": float(mp.lambda_minus),
                "mp_lambda_plus": float(mp.lambda_plus),
                "mp_aspect_ratio": float(mp.aspect_ratio),
                "bulk_outlier_count": bulk_rank,
                "is_mp_outlier": bool(eigenvalue > mp.lambda_plus),
            }
        )
    return rows


def _outlier_loading_frame(
    universe: str,
    empirical_corr,
    sample_size,
    sample_frame,
    estimation,
    method: str,
    *,
    selection_mode: str,
    cumulative_variance_pct: float,
    top_n: int,
) -> pd.DataFrame:
    cleaned, eigenvalues, eigenvectors, mp = _cleaned_eigendecomposition(empirical_corr, sample_size, sample_frame, estimation, method)
    positions = _select_eigen_positions(
        eigenvalues,
        mp,
        selection_mode=selection_mode,
        cumulative_variance_pct=cumulative_variance_pct,
        top_n=top_n,
    )
    if not positions:
        return pd.DataFrame()
    metadata = _sector_metadata(universe, list(cleaned.index))
    tickers_sorted = list(metadata.index)
    reorder = [cleaned.index.get_loc(ticker) for ticker in tickers_sorted]
    index = pd.MultiIndex.from_arrays(
        [
            metadata.loc[tickers_sorted, "sector"].to_numpy(),
            metadata.loc[tickers_sorted, "sub_sector"].to_numpy(),
            np.asarray(tickers_sorted, dtype=object),
        ],
        names=["sector", "sub_sector", "ticker"],
    )
    columns: dict[str, np.ndarray] = {}
    for pos in positions:
        rank = pos + 1
        eigenvalue = eigenvalues[pos]
        columns[f"window_{int(estimation.covariance_window or 0)}_rank_{rank}_lambda_{eigenvalue:.4f}"] = eigenvectors[reorder, pos]
    return pd.DataFrame(columns, index=index)


def _outlier_sector_frames(
    universe: str,
    empirical_corr,
    sample_size,
    sample_frame,
    estimation,
    method: str,
    *,
    selection_mode: str,
    cumulative_variance_pct: float,
    top_n: int,
):
    cleaned, eigenvalues, eigenvectors, mp = _cleaned_eigendecomposition(empirical_corr, sample_size, sample_frame, estimation, method)
    positions = _select_eigen_positions(
        eigenvalues,
        mp,
        selection_mode=selection_mode,
        cumulative_variance_pct=cumulative_variance_pct,
        top_n=top_n,
    )
    if not positions:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    metadata = _sector_metadata(universe, list(cleaned.index))
    tickers_sorted = list(metadata.index)
    reorder = [cleaned.index.get_loc(ticker) for ticker in tickers_sorted]
    abs_columns: dict[str, pd.Series] = {}
    signed_columns: dict[str, pd.Series] = {}
    abs_sub_columns: dict[str, pd.Series] = {}
    signed_sub_columns: dict[str, pd.Series] = {}
    for pos in positions:
        rank = pos + 1
        eigenvalue = eigenvalues[pos]
        column_name = f"window_{int(estimation.covariance_window or 0)}_rank_{rank}_lambda_{eigenvalue:.4f}"
        loading_frame = metadata.copy()
        loading_frame["signed_loading"] = eigenvectors[reorder, pos]
        loading_frame["abs_loading"] = np.abs(loading_frame["signed_loading"])
        loading_frame["sector_sub_sector"] = loading_frame["sector"] + " | " + loading_frame["sub_sector"]
        abs_sector = loading_frame.groupby("sector")["abs_loading"].sum().sort_index()
        abs_total = float(abs_sector.sum())
        if abs_total > 0.0:
            abs_sector = 100.0 * abs_sector / abs_total
        abs_columns[column_name] = abs_sector
        signed_columns[column_name] = loading_frame.groupby("sector")["signed_loading"].sum().sort_index()
        abs_sub = loading_frame.groupby("sector_sub_sector")["abs_loading"].sum().sort_index()
        abs_sub_total = float(abs_sub.sum())
        if abs_sub_total > 0.0:
            abs_sub = 100.0 * abs_sub / abs_sub_total
        abs_sub_columns[column_name] = abs_sub
        signed_sub_columns[column_name] = loading_frame.groupby("sector_sub_sector")["signed_loading"].sum().sort_index()
    return (
        pd.DataFrame(abs_columns).sort_index(),
        pd.DataFrame(signed_columns).sort_index(),
        pd.DataFrame(abs_sub_columns).sort_index(),
        pd.DataFrame(signed_sub_columns).sort_index(),
    )


def _render_window_scree(frame: pd.DataFrame, output_path: Path, *, method: str, matrix_date: pd.Timestamp, log_scale: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    windows = sorted(frame["covariance_window"].unique())
    cmap = plt.get_cmap("tab10", len(windows))
    markers = ["s", "x", "^", "D", "P", "v", "*", "o", "+", "<"]
    for idx, window in enumerate(windows):
        panel = frame.loc[frame["covariance_window"] == window].sort_values("rank")
        color = cmap(idx)
        marker = markers[idx % len(markers)]
        ax.plot(
            panel["rank"],
            panel["eigenvalue"],
            linewidth=2.0,
            color=color,
            label=(
                f"window={window} "
                f"(q={panel['mp_aspect_ratio'].iloc[0]:.2f}, "
                f"lambda+={panel['mp_lambda_plus'].iloc[0]:.2f}, "
                f"outliers={int(panel['bulk_outlier_count'].iloc[0])})"
            ),
        )
        signal_panel = panel.loc[panel["is_mp_outlier"]]
        if not signal_panel.empty:
            ax.scatter(signal_panel["rank"], signal_panel["eigenvalue"], color=color, marker=marker, s=42, linewidths=1.2, zorder=4)
        ax.axhline(panel["mp_lambda_plus"].iloc[0], color=color, linestyle="--", linewidth=1.2, alpha=0.9)
        ax.axhline(panel["mp_lambda_minus"].iloc[0], color=color, linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_title(f"{method} cleaned spectrum by covariance window ({matrix_date.date()})")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _render_window_eigenvalue_distribution(frame: pd.DataFrame, output_path: Path, *, method: str, matrix_date: pd.Timestamp, log_scale: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    windows = sorted(frame["covariance_window"].unique())
    cmap = plt.get_cmap("tab10", len(windows))
    all_eigenvalues = frame["eigenvalue"].to_numpy(dtype=float)
    bins = min(24, max(10, int(np.sqrt(len(all_eigenvalues)))))
    ymax = 0.0
    for idx, window in enumerate(windows):
        panel = frame.loc[frame["covariance_window"] == window].sort_values("rank")
        color = cmap(idx)
        eigenvalues = panel["eigenvalue"].to_numpy(dtype=float)
        mp = marchenko_pastur_law(int(panel["num_assets"].iloc[0]), int(panel["sample_size"].iloc[0]), variance=1.0)
        grid, density = mp.density_grid(num_points=512, padding=0.08)
        hist_density, _, _ = ax.hist(eigenvalues, bins=bins, density=True, alpha=0.18, color=color, edgecolor=color, linewidth=0.8, label=f"empirical window={window}")
        ymax = max(ymax, float(np.max(hist_density)) if len(hist_density) else 0.0, float(np.max(density)) if len(density) else 0.0)
        ax.plot(grid, density, color=color, linewidth=2.0, linestyle="--", label=f"MP window={window}")
        outliers = panel.loc[panel["is_mp_outlier"], "eigenvalue"].to_numpy(dtype=float)
        if len(outliers):
            line_top = ymax * 0.58 if ymax > 0.0 else 0.5
            for eigenvalue in outliers:
                ax.vlines(eigenvalue, ymin=0.0, ymax=line_top, color=color, linestyle=":", linewidth=1.2, alpha=0.9)
    label_y = ymax * 0.62 if ymax > 0.0 else 0.6
    for idx, window in enumerate(windows):
        panel = frame.loc[frame["covariance_window"] == window].sort_values("rank")
        color = cmap(idx)
        outliers = panel.loc[panel["is_mp_outlier"], "eigenvalue"].to_numpy(dtype=float)
        if len(outliers):
            for eigenvalue in outliers:
                ax.text(eigenvalue, label_y, f"{eigenvalue:.2f}", color=color, fontsize=8, rotation=0, ha="center", va="bottom")
    ax.set_ylim(top=max(label_y * 1.15, ymax * 1.02 if ymax > 0.0 else 1.0))
    ax.set_title(f"{method} eigenvalue distribution vs Marchenko-Pastur ({matrix_date.date()})")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    if log_scale:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _render_window_explained_variance(frame: pd.DataFrame, output_path: Path, *, method: str, matrix_date: pd.Timestamp) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    windows = sorted(frame["covariance_window"].unique())
    cmap = plt.get_cmap("tab10", len(windows))
    for idx, window in enumerate(windows):
        panel = frame.loc[frame["covariance_window"] == window].sort_values("rank")
        color = cmap(idx)
        ax.plot(panel["rank"], 100.0 * panel["cumulative_variance_share"], linewidth=2.0, color=color, label=f"window={window}")
    ax.set_title(f"{method} cumulative explained variance by eigenvalue rank ({matrix_date.date()})")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_ylim(0.0, 102.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _render_outlier_sector_presence(abs_frame: pd.DataFrame, output_path: Path, *, method: str, matrix_date: pd.Timestamp) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if abs_frame.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"{method} MP-outlier sector presence ({matrix_date.date()})")
        ax.text(0.5, 0.5, "No MP outlier eigenvectors", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path
    plot_frame = abs_frame.fillna(0.0)
    sectors = list(plot_frame.index)
    columns = list(plot_frame.columns)
    x = np.arange(len(columns))
    bottoms = np.zeros(len(columns), dtype=float)
    cmap = plt.get_cmap("tab20", max(len(sectors), 1))
    fig_width = max(10, 1.6 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    for idx, sector in enumerate(sectors):
        values = plot_frame.loc[sector].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottoms, color=cmap(idx), width=0.72, label=sector)
        bottoms += values
    ax.set_title(f"{method} sector presence in MP-outlier eigenvectors ({matrix_date.date()})")
    ax.set_ylabel("Abs loading share (%)")
    ax.set_xlabel("Eigenvector")
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_ylim(0.0, max(102.0, float(np.max(bottoms)) * 1.02 if len(bottoms) else 102.0))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Sector", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def run_eigenvector_inspection(request: EigenvectorInspectionRequest) -> EigenvectorInspectionResult:
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
    windows = request.windows or list(DEFAULT_WINDOWS)
    windows = parse_windows(",".join(str(item) for item in windows))
    if request.selection_top_n <= 0:
        raise ValueError("selection_top_n must be strictly positive.")
    if request.selection_cumulative_variance <= 0.0 or request.selection_cumulative_variance > 100.0:
        raise ValueError("selection_cumulative_variance must be in (0, 100].")
    if request.method not in supported_cleaning_methods():
        raise ValueError(f"Unknown cleaning method {request.method!r}.")

    prices, quality_report_obj = load_filtered_prices_for_universe(
        universe,
        evaluation_start=evaluation.evaluation_start,
        refresh_policy=request.refresh_policy,
    )
    quality_report = asdict(quality_report_obj)
    target_dates = resolve_target_dates(prices, evaluation)
    if len(target_dates) == 0:
        raise ValueError("No evaluation rebalance dates available for the spectral comparison window.")
    matrix_date = pd.Timestamp(request.matrix_date) if request.matrix_date is not None else target_dates[-1]

    rows: list[dict[str, float | int | str]] = []
    sector_presence_frames: list[pd.DataFrame] = []
    signed_sector_frames: list[pd.DataFrame] = []
    sub_sector_presence_frames: list[pd.DataFrame] = []
    signed_sub_sector_frames: list[pd.DataFrame] = []
    detailed_loading_frames: list[pd.DataFrame] = []
    for window in windows:
        window_estimation = resolve_window_estimation_cfg(estimation, window, min_periods_mode=request.min_periods_mode)
        empirical_corr, sample_size, sample_frame = matrix_sample(prices, window_estimation, matrix_date)
        rows.extend(_spectrum_rows(empirical_corr, sample_size, sample_frame, window_estimation, request.method, matrix_date=matrix_date))
        sector_presence_frame, signed_sector_frame, sub_sector_presence_frame, signed_sub_sector_frame = _outlier_sector_frames(
            universe.name,
            empirical_corr,
            sample_size,
            sample_frame,
            window_estimation,
            request.method,
            selection_mode=request.selection_mode,
            cumulative_variance_pct=request.selection_cumulative_variance,
            top_n=request.selection_top_n,
        )
        loading_frame = _outlier_loading_frame(
            universe.name,
            empirical_corr,
            sample_size,
            sample_frame,
            window_estimation,
            request.method,
            selection_mode=request.selection_mode,
            cumulative_variance_pct=request.selection_cumulative_variance,
            top_n=request.selection_top_n,
        )
        if not sector_presence_frame.empty:
            sector_presence_frames.append(sector_presence_frame)
        if not signed_sector_frame.empty:
            signed_sector_frames.append(signed_sector_frame)
        if not sub_sector_presence_frame.empty:
            sub_sector_presence_frames.append(sub_sector_presence_frame)
        if not signed_sub_sector_frame.empty:
            signed_sub_sector_frames.append(signed_sub_sector_frame)
        if not loading_frame.empty:
            detailed_loading_frames.append(loading_frame)

    scree_frame = pd.DataFrame(rows).sort_values(["covariance_window", "rank"]).reset_index(drop=True)
    sector_presence = pd.concat(sector_presence_frames, axis=1) if sector_presence_frames else pd.DataFrame()
    sector_signed = pd.concat(signed_sector_frames, axis=1) if signed_sector_frames else pd.DataFrame()
    sub_sector_presence = pd.concat(sub_sector_presence_frames, axis=1) if sub_sector_presence_frames else pd.DataFrame()
    sub_sector_signed = pd.concat(signed_sub_sector_frames, axis=1) if signed_sub_sector_frames else pd.DataFrame()
    loadings = pd.concat(detailed_loading_frames, axis=1) if detailed_loading_frames else pd.DataFrame()

    outdir = ensure_output_dir(request.output_dir or "output/optimal_tf/inspection/eigenvectors")
    files: dict[str, Path] = {}
    if outdir is not None:
        selection_suffix = _selection_suffix(
            request.selection_mode,
            cumulative_variance_pct=request.selection_cumulative_variance,
            top_n=request.selection_top_n,
        )
        scree_csv = outdir / "window_method_scree.csv"
        distribution_plot = outdir / "window_method_distribution_mp.png"
        scree_plot = outdir / "window_method_scree.png"
        explained_plot = outdir / "window_method_explained_variance.png"
        sector_presence_csv = outdir / f"window_method_{selection_suffix}_sector_presence.csv"
        sector_signed_csv = outdir / f"window_method_{selection_suffix}_sector_signed.csv"
        sub_sector_presence_csv = outdir / f"window_method_{selection_suffix}_sub_sector_presence.csv"
        sub_sector_signed_csv = outdir / f"window_method_{selection_suffix}_sub_sector_signed.csv"
        loadings_csv = outdir / f"window_method_{selection_suffix}_loadings.csv"
        sector_presence_plot = outdir / f"window_method_{selection_suffix}_sector_presence.png"
        scree_frame.to_csv(scree_csv, index=False)
        sector_presence.to_csv(sector_presence_csv)
        sector_signed.to_csv(sector_signed_csv)
        sub_sector_presence.to_csv(sub_sector_presence_csv)
        sub_sector_signed.to_csv(sub_sector_signed_csv)
        loadings.to_csv(loadings_csv)
        _render_window_scree(scree_frame, scree_plot, method=request.method, matrix_date=matrix_date, log_scale=request.log_scale)
        _render_window_eigenvalue_distribution(scree_frame, distribution_plot, method=request.method, matrix_date=matrix_date, log_scale=request.log_scale)
        _render_window_explained_variance(scree_frame, explained_plot, method=request.method, matrix_date=matrix_date)
        _render_outlier_sector_presence(sector_presence, sector_presence_plot, method=request.method, matrix_date=matrix_date)
        summary = {
            "universe": universe.name,
            "matrix_date": matrix_date.strftime("%Y-%m-%d"),
            "method": request.method,
            "windows": windows,
            "selection_mode": request.selection_mode,
            "selection_cumulative_variance": request.selection_cumulative_variance,
            "selection_top_n": request.selection_top_n,
            "log_scale": bool(request.log_scale),
            "quality_report": quality_report,
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        req = write_request_json(outdir, request)
        files = write_quality_artifacts(outdir, quality_report)
        files.update({
            "scree_csv": scree_csv,
            "scree_plot": scree_plot,
            "distribution_plot": distribution_plot,
            "explained_variance_plot": explained_plot,
            "sector_presence_csv": sector_presence_csv,
            "sector_presence_plot": sector_presence_plot,
            "sector_signed_csv": sector_signed_csv,
            "sub_sector_presence_csv": sub_sector_presence_csv,
            "sub_sector_signed_csv": sub_sector_signed_csv,
            "loadings_csv": loadings_csv,
            "summary": outdir / "summary.json",
        })
        if req is not None:
            files["request"] = req

    return EigenvectorInspectionResult(
        request=request,
        universe=universe.name,
        matrix_date=matrix_date,
        method=request.method,
        windows=windows,
        scree_frame=scree_frame,
        sector_presence=sector_presence,
        sector_signed=sector_signed,
        sub_sector_presence=sub_sector_presence,
        sub_sector_signed=sub_sector_signed,
        loadings=loadings,
        quality_report=quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )


def _sorted_metadata(universe: str, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    metadata = _sector_metadata(universe, tickers)
    sorted_tickers = list(metadata.index)
    return metadata, sorted_tickers


def _sorted_matrix(frame: pd.DataFrame, sorted_tickers: list[str]) -> pd.DataFrame:
    tickers = [ticker for ticker in sorted_tickers if ticker in frame.index and ticker in frame.columns]
    return frame.loc[tickers, tickers]


def _aggregate_matrix_by_groups(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    level: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = [ticker for ticker in metadata.index if ticker in frame.index and ticker in frame.columns]
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ordered = frame.loc[tickers, tickers]
    metadata_slice = metadata.loc[tickers].copy()
    if level == "sector":
        metadata_slice["group_label"] = metadata_slice["sector"].astype(str)
        membership = (
            metadata_slice.reset_index()
            .rename(columns={"index": "ticker"})
            .loc[:, ["group_label", "sector", "ticker"]]
            .rename(columns={"group_label": "group"})
        )
    elif level == "sub_sector":
        metadata_slice["group_label"] = (metadata_slice["sector"].astype(str) + " | " + metadata_slice["sub_sector"].astype(str)).astype(str)
        membership = (
            metadata_slice.reset_index()
            .rename(columns={"index": "ticker"})
            .loc[:, ["group_label", "sector", "sub_sector", "ticker"]]
            .rename(columns={"group_label": "group"})
        )
    else:
        raise ValueError(f"Unknown aggregation level {level!r}.")

    membership_counts = (
        membership.groupby("group", sort=False)
        .size()
        .rename("num_tickers")
        .reset_index()
    )
    membership = membership.merge(membership_counts, on="group", how="left")
    if level == "sector":
        membership = membership.loc[:, ["group", "sector", "num_tickers", "ticker"]]
    else:
        membership = membership.loc[:, ["group", "sector", "sub_sector", "num_tickers", "ticker"]]

    group_labels = membership_counts["group"].tolist()
    label_array = metadata_slice["group_label"].to_numpy(dtype=object)
    values = ordered.to_numpy(dtype=float)
    aggregated = pd.DataFrame(index=group_labels, columns=group_labels, dtype=float)
    pair_counts = pd.DataFrame(index=group_labels, columns=group_labels, dtype=float)

    masks = {group: (label_array == group) for group in group_labels}
    for left_group in group_labels:
        left_mask = masks[left_group]
        left_count = int(np.sum(left_mask))
        for right_group in group_labels:
            right_mask = masks[right_group]
            block = values[np.ix_(left_mask, right_mask)]
            if left_group == right_group and left_count > 1:
                block_values = block[~np.eye(left_count, dtype=bool)]
            else:
                block_values = block.reshape(-1)
            finite = block_values[np.isfinite(block_values)]
            pair_counts.loc[left_group, right_group] = float(finite.size)
            aggregated.loc[left_group, right_group] = float(finite.mean()) if finite.size else np.nan

    return aggregated, pair_counts, membership


def _equal_weight_group_correlation(
    returns_frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    level: str,
) -> pd.DataFrame:
    tickers = [ticker for ticker in metadata.index if ticker in returns_frame.columns]
    if not tickers:
        return pd.DataFrame()

    metadata_slice = metadata.loc[tickers].copy()
    if level == "sector":
        metadata_slice["group_label"] = metadata_slice["sector"].astype(str)
    elif level == "sub_sector":
        metadata_slice["group_label"] = (
            metadata_slice["sector"].astype(str) + " | " + metadata_slice["sub_sector"].astype(str)
        ).astype(str)
    else:
        raise ValueError(f"Unknown aggregation level {level!r}.")

    group_labels = pd.Index(pd.unique(metadata_slice["group_label"]), dtype=object).tolist()
    ew_returns: dict[str, pd.Series] = {}
    for group_label in group_labels:
        group_tickers = metadata_slice.index[metadata_slice["group_label"] == group_label].tolist()
        group_frame = returns_frame.loc[:, group_tickers]
        ew_returns[str(group_label)] = group_frame.mean(axis=1, skipna=True)
    ew_frame = pd.DataFrame(ew_returns).dropna(axis=1, how="all")
    if ew_frame.empty:
        return pd.DataFrame()
    return ew_frame.corr()


def _spectrum_frame(eigenvalues: np.ndarray, *, label: str) -> pd.DataFrame:
    total = float(np.sum(eigenvalues))
    cumulative = np.cumsum(eigenvalues)
    return pd.DataFrame(
        {
            "matrix": label,
            "rank": np.arange(1, len(eigenvalues) + 1, dtype=int),
            "eigenvalue": eigenvalues.astype(float),
            "variance_share": np.where(total > 0.0, eigenvalues / total, 0.0),
            "cumulative_variance_share": np.where(total > 0.0, cumulative / total, 0.0),
        }
    )


def _eigenvector_frame(
    matrix: pd.DataFrame,
    *,
    universe: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.to_numpy(dtype=float))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    metadata, sorted_tickers = _sorted_metadata(universe, list(matrix.index))
    reorder = [matrix.index.get_loc(ticker) for ticker in sorted_tickers]
    index = pd.MultiIndex.from_arrays(
        [
            metadata.loc[sorted_tickers, "sector"].to_numpy(),
            metadata.loc[sorted_tickers, "sub_sector"].to_numpy(),
            np.asarray(sorted_tickers, dtype=object),
        ],
        names=["sector", "sub_sector", "ticker"],
    )
    columns = [f"{prefix}{rank}" for rank in range(1, len(eigenvalues) + 1)]
    vectors = pd.DataFrame(eigenvectors[reorder, :], index=index, columns=columns)
    return _spectrum_frame(eigenvalues, label=prefix.rstrip("_")), vectors


def _spectral_frame_from_decomposition(
    decomposition,
    *,
    universe: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eigenvalues = np.asarray(decomposition.eigenvalues, dtype=float)
    eigenvectors = decomposition.eigenvectors.to_numpy(dtype=float)
    tickers = list(decomposition.eigenvectors.index)
    metadata, sorted_tickers = _sorted_metadata(universe, tickers)
    reorder = [tickers.index(ticker) for ticker in sorted_tickers]
    index = pd.MultiIndex.from_arrays(
        [
            metadata.loc[sorted_tickers, "sector"].to_numpy(),
            metadata.loc[sorted_tickers, "sub_sector"].to_numpy(),
            np.asarray(sorted_tickers, dtype=object),
        ],
        names=["sector", "sub_sector", "ticker"],
    )
    columns = [f"{prefix}{rank}" for rank in range(1, len(eigenvalues) + 1)]
    vectors = pd.DataFrame(eigenvectors[reorder, :], index=index, columns=columns)
    return _spectrum_frame(eigenvalues, label=prefix.rstrip("_")), vectors


_EIGENPORTFOLIO_MIN_WEIGHT_SUM = 1e-8
_EIGENPORTFOLIO_EXTRA_RANKS_AFTER_MP = 2


def _normalize_eigenportfolio_matrix_by_weight_sum(
    eigenvector_frame: pd.DataFrame,
    *,
    min_abs_weight_sum: float = _EIGENPORTFOLIO_MIN_WEIGHT_SUM,
) -> pd.DataFrame:
    if eigenvector_frame.empty:
        return eigenvector_frame.copy()
    weights = eigenvector_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    weight_sums = weights.sum(axis=0)
    valid_columns = weight_sums.index[weight_sums.abs() > float(min_abs_weight_sum)]
    if len(valid_columns) == 0:
        return weights.iloc[:, 0:0].copy()
    normalized = weights.loc[:, valid_columns].divide(weight_sums.loc[valid_columns], axis=1)
    return normalized.fillna(0.0)


def _selected_eigenportfolio_columns(
    normalized_weight_matrix: pd.DataFrame,
    spectrum_frame: pd.DataFrame,
    *,
    num_assets: int,
    sample_size: int,
    extra_ranks_after_mp: int = _EIGENPORTFOLIO_EXTRA_RANKS_AFTER_MP,
) -> list[str]:
    if normalized_weight_matrix.empty or spectrum_frame.empty:
        return []
    try:
        mp_law = marchenko_pastur_law(num_assets=num_assets, sample_size=sample_size, variance=1.0)
    except ValueError:
        return []
    ordered = spectrum_frame.copy()
    ordered["rank"] = pd.to_numeric(ordered["rank"], errors="coerce")
    ordered["eigenvalue"] = pd.to_numeric(ordered["eigenvalue"], errors="coerce")
    ordered = ordered.dropna(subset=["rank", "eigenvalue"]).sort_values("rank").reset_index(drop=True)
    if ordered.empty:
        return []
    outlier_mask = ordered["eigenvalue"] > float(mp_law.lambda_plus)
    outlier_count = int(outlier_mask.sum())
    selection_count = min(len(ordered), outlier_count + max(0, int(extra_ranks_after_mp)))
    if selection_count <= 0:
        selection_count = min(len(ordered), max(1, int(extra_ranks_after_mp)))
    selected: list[str] = []
    for _, spectrum_row in ordered.iloc[:selection_count].iterrows():
        column_name = f"corr_ev{int(spectrum_row['rank'])}"
        if column_name in normalized_weight_matrix.columns:
            selected.append(column_name)
    return selected


def _build_eigenportfolio_outputs(
    returns_frame: pd.DataFrame,
    eigenvector_frame: pd.DataFrame,
    spectrum_frame: pd.DataFrame,
    *,
    num_assets: int,
    sample_size: int,
    extra_ranks_after_mp: int = _EIGENPORTFOLIO_EXTRA_RANKS_AFTER_MP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if returns_frame.empty or eigenvector_frame.empty or spectrum_frame.empty:
        empty = pd.DataFrame(index=returns_frame.index.copy())
        return eigenvector_frame.iloc[:, 0:0].copy(), empty, pd.DataFrame()
    tickers = [str(item[-1]) if isinstance(item, tuple) else str(item) for item in eigenvector_frame.index]
    aligned_returns = returns_frame.reindex(columns=tickers).fillna(0.0)
    normalized_weight_matrix = _normalize_eigenportfolio_matrix_by_weight_sum(eigenvector_frame)
    if normalized_weight_matrix.empty:
        return normalized_weight_matrix, pd.DataFrame(index=returns_frame.index.copy()), pd.DataFrame()
    all_component_returns = aligned_returns.to_numpy(dtype=float) @ normalized_weight_matrix.to_numpy(dtype=float)
    all_component_returns = pd.DataFrame(
        all_component_returns,
        index=aligned_returns.index,
        columns=list(normalized_weight_matrix.columns),
    )
    selected_column_names = _selected_eigenportfolio_columns(
        normalized_weight_matrix,
        spectrum_frame,
        num_assets=num_assets,
        sample_size=sample_size,
        extra_ranks_after_mp=extra_ranks_after_mp,
    )
    if not selected_column_names:
        return normalized_weight_matrix.iloc[:, 0:0].copy(), pd.DataFrame(index=aligned_returns.index), pd.DataFrame()

    selected_weights = normalized_weight_matrix.loc[:, selected_column_names].copy()
    selected_returns = all_component_returns.loc[:, selected_column_names].copy()
    selected_nav = pd.DataFrame(
        {column_name: cumulative_nav(selected_returns[column_name]).rename(column_name) for column_name in selected_column_names},
        index=selected_returns.index,
    )

    summary_rows: list[dict[str, float | str | int]] = []
    for column_name in selected_column_names:
        returns_series = selected_returns[column_name]
        metrics = evaluation_metrics(
            returns_series,
            pd.Series(0.0, index=returns_series.index, dtype=float),
            pd.Series(0.0, index=returns_series.index, dtype=float),
            num_rebalances=0,
        )
        rank = int(str(column_name).removeprefix("corr_ev"))
        spectrum_match = spectrum_frame.loc[pd.to_numeric(spectrum_frame["rank"], errors="coerce") == rank]
        eigenvalue = float(pd.to_numeric(spectrum_match["eigenvalue"].iloc[0], errors="coerce")) if not spectrum_match.empty else float("nan")
        variance_share = float(pd.to_numeric(spectrum_match["variance_share"].iloc[0], errors="coerce")) if not spectrum_match.empty else float("nan")
        cumulative_share = float(pd.to_numeric(spectrum_match["cumulative_variance_share"].iloc[0], errors="coerce")) if not spectrum_match.empty else float("nan")
        summary_rows.append(
            {
                "eigenportfolio": column_name,
                "rank": rank,
                "eigenvalue": eigenvalue,
                "variance_share": variance_share,
                "cumulative_variance_share": cumulative_share,
                "cagr": float(metrics.cagr),
                "ann_vol": float(metrics.ann_vol),
                "sharpe": float(metrics.sharpe),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    return selected_weights, selected_nav, summary_frame


def run_inspection_snapshot(request: InspectionSnapshotRequest) -> InspectionSnapshotResult:
    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(request.config_path)
    del allocation, compare, output

    universe, estimation, backtest, evaluation = _resolve_config_overrides(
        request,
        default_universe=universe,
        default_estimation=estimation,
        default_backtest=backtest,
        default_evaluation=evaluation,
    )

    cleaning_method = _resolve_inspection_cleaning_method(request.cleaning_method, fallback=estimation.cleaning_method)
    input_type = _resolve_input_type(request.input_type)
    matrix_type = _resolve_matrix_type(request.matrix_type)
    estimator_method = _resolve_estimator_method(request.estimator_method)
    estimator_window = _resolve_estimator_window(request, estimation)
    snapshot_estimation = resolve_window_estimation_cfg(
        estimation,
        estimator_window,
        min_periods_mode="clamp",
    )
    snapshot_estimation = snapshot_estimation.__class__(**{
        **snapshot_estimation.__dict__,
        "cleaning_method": cleaning_method,
        "linear_shrinkage": _resolve_linear_shrinkage_intensity(request, snapshot_estimation),
    })
    prices, quality_report_obj = load_filtered_prices_for_universe(
        universe,
        evaluation_start=evaluation.evaluation_start,
        refresh_policy=request.refresh_policy,
    )
    quality_report = asdict(quality_report_obj)
    allocation_date = resolve_allocation_date(prices.index, as_of_date=request.date)
    history = prices.loc[prices.index <= allocation_date]
    if history.empty:
        raise ValueError(f"No price history available on or before {allocation_date.date()}.")

    empirical_corr, sample_cov, sample_size, sample_frame = matrix_sample_bundle(
        history,
        snapshot_estimation,
        allocation_date,
        input_type=_matrix_sample_input_mode(input_type),
        estimator_method=_matrix_sample_estimator_mode(estimator_method),
        estimator_window=estimator_window,
    )
    empirical_cleaner = clean_correlation_matrix_rich(
        empirical_corr,
        data=sample_frame,
        sample_size=sample_size,
        method="empirical",
        linear_shrinkage=0.0,
        bandwidth=snapshot_estimation.rie_bandwidth,
    )
    selected_cleaner = clean_correlation_matrix_rich(
        empirical_corr,
        data=sample_frame,
        sample_size=sample_size,
        method=cleaning_method,
        linear_shrinkage=snapshot_estimation.linear_shrinkage,
        bandwidth=snapshot_estimation.rie_bandwidth,
    )
    empirical_cleaned_corr = empirical_cleaner.cleaned_matrix
    cleaned_corr = selected_cleaner.cleaned_matrix

    returns, vol, z_returns = sanitized_normalized_returns(history, snapshot_estimation)
    trend = trend_ema_signal(z_returns, alpha=snapshot_estimation.trend_alpha, span=snapshot_estimation.trend_span)
    sample_vol = pd.Series(
        np.sqrt(np.clip(np.diag(sample_cov.to_numpy(dtype=float)), 0.0, None)),
        index=sample_cov.index,
        dtype=float,
    ).reindex(cleaned_corr.index).fillna(0.0)
    empirical_cleaned_cov = correlation_to_covariance(empirical_cleaned_corr, sample_vol)
    cleaned_cov = correlation_to_covariance(cleaned_corr, sample_vol)

    metadata, sorted_tickers = _sorted_metadata(universe.name, list(cleaned_corr.index))
    sorted_sample_corr = _sorted_matrix(empirical_corr, sorted_tickers)
    sorted_sample_cov = _sorted_matrix(sample_cov, sorted_tickers)
    sorted_empirical_cleaned_corr = _sorted_matrix(empirical_cleaned_corr, sorted_tickers)
    sorted_empirical_cleaned_cov = _sorted_matrix(empirical_cleaned_cov, sorted_tickers)
    sorted_cleaned_corr = _sorted_matrix(cleaned_corr, sorted_tickers)
    sorted_cleaned_cov = _sorted_matrix(cleaned_cov, sorted_tickers)
    sample_sector_matrix, sector_pair_counts, sector_membership = _aggregate_matrix_by_groups(
        sorted_sample_cov if matrix_type == "covariance" else sorted_sample_corr,
        metadata,
        level="sector",
    )
    empirical_cleaned_sector_matrix, _, _ = _aggregate_matrix_by_groups(
        sorted_empirical_cleaned_cov if matrix_type == "covariance" else sorted_empirical_cleaned_corr,
        metadata,
        level="sector",
    )
    cleaned_sector_matrix, _, _ = _aggregate_matrix_by_groups(
        sorted_cleaned_cov if matrix_type == "covariance" else sorted_cleaned_corr,
        metadata,
        level="sector",
    )
    sample_sector_ew_correlation = _equal_weight_group_correlation(
        sample_frame.reindex(columns=sorted_tickers),
        metadata,
        level="sector",
    )
    sample_sub_sector_matrix, sub_sector_pair_counts, sub_sector_membership = _aggregate_matrix_by_groups(
        sorted_sample_cov if matrix_type == "covariance" else sorted_sample_corr,
        metadata,
        level="sub_sector",
    )
    empirical_cleaned_sub_sector_matrix, _, _ = _aggregate_matrix_by_groups(
        sorted_empirical_cleaned_cov if matrix_type == "covariance" else sorted_empirical_cleaned_corr,
        metadata,
        level="sub_sector",
    )
    cleaned_sub_sector_matrix, _, _ = _aggregate_matrix_by_groups(
        sorted_cleaned_cov if matrix_type == "covariance" else sorted_cleaned_corr,
        metadata,
        level="sub_sector",
    )

    corr_spectrum, corr_eigenvectors = _spectral_frame_from_decomposition(
        selected_cleaner.cleaned,
        universe=universe.name,
        prefix="corr_ev",
    )
    cov_spectrum, cov_eigenvectors = _eigenvector_frame(sorted_cleaned_cov, universe=universe.name, prefix="cov_ev")

    feature_index = pd.MultiIndex.from_arrays(
        [
            metadata.loc[sorted_tickers, "sector"].to_numpy(),
            metadata.loc[sorted_tickers, "sub_sector"].to_numpy(),
            np.asarray(sorted_tickers, dtype=object),
        ],
        names=["sector", "sub_sector", "ticker"],
    )
    last_prices = history.loc[:allocation_date].ffill().iloc[-1].reindex(sorted_tickers)
    feature_frame = pd.DataFrame(
        {
            "last_price": last_prices.to_numpy(dtype=float),
            "last_return": returns.loc[:allocation_date].iloc[-1].reindex(sorted_tickers).to_numpy(dtype=float),
            "ewma_vol": vol.loc[:allocation_date].ffill().iloc[-1].reindex(sorted_tickers).to_numpy(dtype=float),
            "z_return": z_returns.loc[:allocation_date].fillna(0.0).iloc[-1].reindex(sorted_tickers).to_numpy(dtype=float),
            "trend_signal": trend.loc[:allocation_date].fillna(0.0).iloc[-1].reindex(sorted_tickers).to_numpy(dtype=float),
        },
        index=feature_index,
    )

    corr_diff = sorted_cleaned_corr - sorted_empirical_cleaned_corr
    cleaner_comparison_frame = pd.DataFrame(
        [
            {
                "reference_cleaner": "empirical",
                "selected_cleaner": cleaning_method,
                "linear_shrinkage": float(snapshot_estimation.linear_shrinkage),
                "max_abs_corr_diff_vs_empirical": float(corr_diff.abs().to_numpy().max()),
                "mean_abs_corr_diff_vs_empirical": float(corr_diff.abs().to_numpy().mean()),
                "fro_corr_diff_vs_empirical": float(np.linalg.norm(corr_diff.to_numpy(dtype=float), ord="fro")),
            }
        ]
    )
    sorted_component_input_frame = returns.loc[:allocation_date].reindex(columns=sorted_tickers).fillna(0.0)
    if evaluation.evaluation_start is not None:
        evaluation_start_ts = pd.Timestamp(evaluation.evaluation_start)
        sorted_component_input_frame = sorted_component_input_frame.loc[
            sorted_component_input_frame.index >= evaluation_start_ts
        ]
    correlation_eigenportfolios, correlation_component_nav, correlation_component_summary = _build_eigenportfolio_outputs(
        sorted_component_input_frame,
        corr_eigenvectors,
        corr_spectrum,
        num_assets=int(len(sorted_tickers)),
        sample_size=int(sample_size),
    )
    outdir = ensure_output_dir(request.output_dir or "output/optimal_tf/inspection/snapshot")
    files: dict[str, Path] = {}
    if outdir is not None:
        sample_corr_path = outdir / "sample_correlation.csv"
        sample_cov_path = outdir / "sample_covariance.csv"
        empirical_cleaned_corr_path = outdir / "empirical_cleaned_correlation.csv"
        empirical_cleaned_cov_path = outdir / "empirical_cleaned_covariance.csv"
        cleaned_corr_path = outdir / "cleaned_correlation.csv"
        cleaned_cov_path = outdir / "cleaned_covariance.csv"
        sector_matrix_path = outdir / "cleaned_sector_matrix.csv"
        sector_baseline_path = outdir / "empirical_cleaned_sector_matrix.csv"
        sector_sample_path = outdir / "sample_sector_matrix.csv"
        sector_ew_corr_path = outdir / "sample_sector_ew_correlation.csv"
        sector_counts_path = outdir / "sector_pair_counts.csv"
        sector_membership_path = outdir / "sector_membership.csv"
        sub_sector_matrix_path = outdir / "cleaned_sub_sector_matrix.csv"
        sub_sector_baseline_path = outdir / "empirical_cleaned_sub_sector_matrix.csv"
        sub_sector_sample_path = outdir / "sample_sub_sector_matrix.csv"
        sub_sector_counts_path = outdir / "sub_sector_pair_counts.csv"
        sub_sector_membership_path = outdir / "sub_sector_membership.csv"
        corr_spectrum_path = outdir / "correlation_spectrum.csv"
        cov_spectrum_path = outdir / "covariance_spectrum.csv"
        corr_vectors_path = outdir / "correlation_eigenvectors.csv"
        cov_vectors_path = outdir / "covariance_eigenvectors.csv"
        corr_portfolios_path = outdir / "correlation_eigenportfolios.csv"
        corr_component_nav_path = outdir / "correlation_component_nav.csv"
        corr_component_summary_path = outdir / "correlation_component_summary.csv"
        features_path = outdir / "features.csv"
        cleaner_comparison_path = outdir / "cleaner_comparison.csv"
        sorted_sample_corr.to_csv(sample_corr_path)
        sorted_sample_cov.to_csv(sample_cov_path)
        sorted_empirical_cleaned_corr.to_csv(empirical_cleaned_corr_path)
        sorted_empirical_cleaned_cov.to_csv(empirical_cleaned_cov_path)
        sorted_cleaned_corr.to_csv(cleaned_corr_path)
        sorted_cleaned_cov.to_csv(cleaned_cov_path)
        cleaned_sector_matrix.to_csv(sector_matrix_path)
        empirical_cleaned_sector_matrix.to_csv(sector_baseline_path)
        sample_sector_matrix.to_csv(sector_sample_path)
        sample_sector_ew_correlation.to_csv(sector_ew_corr_path)
        sector_pair_counts.to_csv(sector_counts_path)
        sector_membership.to_csv(sector_membership_path, index=False)
        cleaned_sub_sector_matrix.to_csv(sub_sector_matrix_path)
        empirical_cleaned_sub_sector_matrix.to_csv(sub_sector_baseline_path)
        sample_sub_sector_matrix.to_csv(sub_sector_sample_path)
        sub_sector_pair_counts.to_csv(sub_sector_counts_path)
        sub_sector_membership.to_csv(sub_sector_membership_path, index=False)
        corr_spectrum.to_csv(corr_spectrum_path, index=False)
        cov_spectrum.to_csv(cov_spectrum_path, index=False)
        corr_eigenvectors.to_csv(corr_vectors_path)
        cov_eigenvectors.to_csv(cov_vectors_path)
        correlation_eigenportfolios.to_csv(corr_portfolios_path)
        correlation_component_nav.to_csv(corr_component_nav_path)
        correlation_component_summary.to_csv(corr_component_summary_path, index=False)
        feature_frame.to_csv(features_path)
        cleaner_comparison_frame.to_csv(cleaner_comparison_path, index=False)
        request_path = write_request_json(outdir, request)
        summary_path = write_json(
            outdir,
            "summary.json",
            {
                "universe": universe.name,
                "cleaning_method": cleaning_method,
                "input_type": input_type,
                "matrix_type": matrix_type,
                "estimator_method": estimator_method,
                "estimator_window": estimator_window,
                "allocation_date": allocation_date.strftime("%Y-%m-%d"),
                "num_assets": int(len(sorted_tickers)),
                "sample_size": int(sample_size),
                "cleaner_comparison": cleaner_comparison_frame.iloc[0].to_dict(),
                "quality_report": quality_report,
            },
        )
        files = write_quality_artifacts(outdir, quality_report)
        files.update({
            "sample_correlation": sample_corr_path,
            "sample_covariance": sample_cov_path,
            "empirical_cleaned_correlation": empirical_cleaned_corr_path,
            "empirical_cleaned_covariance": empirical_cleaned_cov_path,
            "cleaned_correlation": cleaned_corr_path,
            "cleaned_covariance": cleaned_cov_path,
            "sample_sector_matrix": sector_sample_path,
            "empirical_cleaned_sector_matrix": sector_baseline_path,
            "cleaned_sector_matrix": sector_matrix_path,
            "sample_sector_ew_correlation": sector_ew_corr_path,
            "sector_pair_counts": sector_counts_path,
            "sector_membership": sector_membership_path,
            "sample_sub_sector_matrix": sub_sector_sample_path,
            "empirical_cleaned_sub_sector_matrix": sub_sector_baseline_path,
            "cleaned_sub_sector_matrix": sub_sector_matrix_path,
            "sub_sector_pair_counts": sub_sector_counts_path,
            "sub_sector_membership": sub_sector_membership_path,
            "correlation_spectrum": corr_spectrum_path,
            "covariance_spectrum": cov_spectrum_path,
            "correlation_eigenvectors": corr_vectors_path,
            "covariance_eigenvectors": cov_vectors_path,
            "correlation_eigenportfolios": corr_portfolios_path,
            "correlation_component_nav": corr_component_nav_path,
            "correlation_component_summary": corr_component_summary_path,
            "features": features_path,
            "cleaner_comparison": cleaner_comparison_path,
        })
        if request_path is not None:
            files["request"] = request_path
        if summary_path is not None:
            files["summary"] = summary_path

    return InspectionSnapshotResult(
        request=request,
        universe=universe.name,
        cleaning_method=cleaning_method,
        input_type=input_type,
        matrix_type=matrix_type,
        estimator_method=estimator_method,
        estimator_window=estimator_window,
        allocation_date=allocation_date,
        sample_size=int(sample_size),
        num_assets=int(len(sorted_tickers)),
        sample_correlation=sorted_sample_corr,
        sample_covariance=sorted_sample_cov,
        empirical_cleaned_correlation=sorted_empirical_cleaned_corr,
        empirical_cleaned_covariance=sorted_empirical_cleaned_cov,
        cleaned_correlation=sorted_cleaned_corr,
        cleaned_covariance=sorted_cleaned_cov,
        sample_sector_matrix=sample_sector_matrix,
        empirical_cleaned_sector_matrix=empirical_cleaned_sector_matrix,
        cleaned_sector_matrix=cleaned_sector_matrix,
        sample_sector_ew_correlation=sample_sector_ew_correlation,
        sector_pair_counts=sector_pair_counts,
        sector_membership=sector_membership,
        sample_sub_sector_matrix=sample_sub_sector_matrix,
        empirical_cleaned_sub_sector_matrix=empirical_cleaned_sub_sector_matrix,
        cleaned_sub_sector_matrix=cleaned_sub_sector_matrix,
        sub_sector_pair_counts=sub_sector_pair_counts,
        sub_sector_membership=sub_sector_membership,
        correlation_spectrum=corr_spectrum,
        covariance_spectrum=cov_spectrum,
        correlation_eigenvectors=corr_eigenvectors,
        covariance_eigenvectors=cov_eigenvectors,
        correlation_eigenportfolios=correlation_eigenportfolios,
        correlation_component_nav=correlation_component_nav,
        correlation_component_summary=correlation_component_summary,
        feature_frame=feature_frame,
        cleaner_comparison_frame=cleaner_comparison_frame,
        quality_report=quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )


def run_inspection_interval(request: InspectionIntervalRequest) -> InspectionIntervalResult:
    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(request.config_path)
    del allocation, compare, output

    universe, estimation, backtest, evaluation = _resolve_config_overrides(
        request,
        default_universe=universe,
        default_estimation=estimation,
        default_backtest=backtest,
        default_evaluation=evaluation,
    )
    del backtest

    cleaning_method = _resolve_inspection_cleaning_method(request.cleaning_method, fallback=estimation.cleaning_method)
    input_type = _resolve_input_type(request.input_type)
    matrix_type = _resolve_matrix_type(request.matrix_type)
    estimator_method = _resolve_estimator_method(request.estimator_method)
    leading_eigenvectors = max(1, int(request.leading_eigenvectors or 3))

    estimator_window = _resolve_estimator_window(request, estimation)
    interval_estimation = resolve_window_estimation_cfg(
        estimation,
        estimator_window,
        min_periods_mode="clamp",
    )
    interval_estimation = interval_estimation.__class__(**{
        **interval_estimation.__dict__,
        "cleaning_method": cleaning_method,
        "linear_shrinkage": _resolve_linear_shrinkage_intensity(request, interval_estimation),
    })

    prices, quality_report_obj = load_filtered_prices_for_universe(
        universe,
        evaluation_start=evaluation.evaluation_start,
        refresh_policy=request.refresh_policy,
    )
    quality_report = asdict(quality_report_obj)
    target_dates = pd.DatetimeIndex(resolve_target_dates(prices, evaluation))
    target_dates = target_dates[target_dates.isin(prices.index)]
    if target_dates.empty:
        raise ValueError("No inspection dates available in the selected interval.")

    summary_rows: list[dict[str, float | int | str]] = []
    spectrum_rows: list[dict[str, float | int | str | bool]] = []
    alignment_rows: list[dict[str, float | int | str | None]] = []
    previous_vectors: pd.DataFrame | None = None
    anchor_vectors: pd.DataFrame | None = None
    num_assets = 0
    skipped_dates: list[str] = []

    for matrix_date in target_dates:
        history = prices.loc[prices.index <= matrix_date]
        if history.empty:
            continue
        try:
            empirical_corr, sample_cov, sample_size, sample_frame = matrix_sample_bundle(
                history,
                interval_estimation,
                matrix_date,
                input_type=_matrix_sample_input_mode(input_type),
                estimator_method=_matrix_sample_estimator_mode(estimator_method),
                estimator_window=estimator_window,
            )
        except ValueError as exc:
            message = str(exc)
            if (
                message.startswith("No correlation sample available on ")
                or message.startswith("No EWMA covariance sample available on ")
            ):
                skipped_dates.append(matrix_date.strftime("%Y-%m-%d"))
                continue
            raise
        cleaned_corr = clean_correlation_matrix(
            empirical_corr,
            data=sample_frame,
            sample_size=sample_size,
            method=cleaning_method,
            linear_shrinkage=interval_estimation.linear_shrinkage,
            bandwidth=interval_estimation.rie_bandwidth,
        )
        sample_vol = pd.Series(
            np.sqrt(np.clip(np.diag(sample_cov.to_numpy(dtype=float)), 0.0, None)),
            index=sample_cov.index,
            dtype=float,
        ).reindex(cleaned_corr.index).fillna(0.0)
        cleaned_cov = correlation_to_covariance(cleaned_corr, sample_vol)
        metadata, sorted_tickers = _sorted_metadata(universe.name, list(cleaned_corr.index))
        del metadata
        sorted_cleaned_corr = _sorted_matrix(cleaned_corr, sorted_tickers)
        sorted_cleaned_cov = _sorted_matrix(cleaned_cov, sorted_tickers)
        selected_matrix = sorted_cleaned_cov if matrix_type == "covariance" else sorted_cleaned_corr

        eigenvalues, eigenvectors = np.linalg.eigh(selected_matrix.to_numpy(dtype=float))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order].astype(float)
        eigenvectors = eigenvectors[:, order].astype(float)
        total = float(np.sum(eigenvalues))
        cumulative = np.cumsum(eigenvalues)
        mp = marchenko_pastur_law(len(sorted_cleaned_corr), sample_size, variance=1.0)
        num_assets = int(len(sorted_tickers))

        bulk_outlier_count = int(np.sum(np.linalg.eigvalsh(sorted_cleaned_corr.to_numpy(dtype=float)) > mp.lambda_plus))
        summary_rows.append(
            {
                "date": matrix_date.strftime("%Y-%m-%d"),
                "sample_size": int(sample_size),
                "num_assets": num_assets,
                "trace": float(np.trace(selected_matrix.to_numpy(dtype=float))),
                "leading_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) else 0.0,
                "second_eigenvalue": float(eigenvalues[1]) if len(eigenvalues) > 1 else np.nan,
                "third_eigenvalue": float(eigenvalues[2]) if len(eigenvalues) > 2 else np.nan,
                "bulk_outlier_count": bulk_outlier_count,
                "mp_lambda_plus": float(mp.lambda_plus),
            }
        )

        for rank, eigenvalue in enumerate(eigenvalues, start=1):
            spectrum_rows.append(
                {
                    "date": matrix_date.strftime("%Y-%m-%d"),
                    "rank": rank,
                    "eigenvalue": float(eigenvalue),
                    "variance_share": float(eigenvalue / total) if total else 0.0,
                    "cumulative_variance_share": float(cumulative[rank - 1] / total) if total else 0.0,
                    "num_assets": num_assets,
                    "sample_size": int(sample_size),
                    "mp_lambda_plus": float(mp.lambda_plus),
                    "is_mp_outlier": bool(rank <= bulk_outlier_count),
                }
            )

        active_components = min(leading_eigenvectors, eigenvectors.shape[1])
        current_vectors = pd.DataFrame(
            eigenvectors[:, :active_components],
            index=sorted_tickers,
            columns=[f"rank_{position + 1}" for position in range(active_components)],
            dtype=float,
        )
        if anchor_vectors is None:
            anchor_vectors = current_vectors.copy()
        for position in range(active_components):
            current_vector = current_vectors.iloc[:, position]
            previous_alignment = None
            if previous_vectors is not None and position < previous_vectors.shape[1]:
                previous_alignment = _absolute_vector_alignment(
                    current_vector,
                    previous_vectors.iloc[:, position],
                )
            anchor_alignment = _absolute_vector_alignment(
                current_vector,
                anchor_vectors.iloc[:, position],
            )
            alignment_rows.append(
                {
                    "date": matrix_date.strftime("%Y-%m-%d"),
                    "rank": position + 1,
                    "abs_alignment_previous": previous_alignment,
                    "abs_alignment_anchor": anchor_alignment,
                }
            )
        previous_vectors = current_vectors.copy()

    if not summary_rows:
        raise ValueError("No matrix inspection data could be computed over the selected interval.")

    summary_frame = pd.DataFrame(summary_rows)
    spectrum_frame = pd.DataFrame(spectrum_rows)
    retained_spectrum_frame = spectrum_frame.loc[
        pd.to_numeric(spectrum_frame["rank"], errors="coerce") <= max(1, leading_eigenvectors)
    ].copy()
    retained_pivot = retained_spectrum_frame.pivot(index="date", columns="rank", values="eigenvalue").sort_index()
    retained_pivot.index = pd.to_datetime(retained_pivot.index)
    retained_pivot.columns = [f"rank_{int(column)}" for column in retained_pivot.columns]
    variogram_lag_max = max(600, 2 * int(num_assets))
    variogram_frame = _variogram_frame(retained_pivot, max_lag=variogram_lag_max)
    eigenvector_similarity_frame = pd.DataFrame(alignment_rows)

    outdir = ensure_output_dir(request.output_dir or "output/optimal_tf/inspection/interval")
    files: dict[str, Path] = {}
    if outdir is not None:
        summary_path = outdir / "summary_frame.csv"
        spectrum_path = outdir / "spectrum_frame.csv"
        variogram_path = outdir / "eigenvalue_variogram.csv"
        alignment_path = outdir / "eigenvector_similarity.csv"
        summary_frame.to_csv(summary_path, index=False)
        spectrum_frame.to_csv(spectrum_path, index=False)
        variogram_frame.to_csv(variogram_path, index=False)
        eigenvector_similarity_frame.to_csv(alignment_path, index=False)
        request_path = write_request_json(outdir, request)
        summary_json_path = write_json(
            outdir,
            "summary.json",
            {
                "universe": universe.name,
                "cleaning_method": cleaning_method,
                "input_type": input_type,
                "matrix_type": matrix_type,
                "estimator_method": estimator_method,
                "estimator_window": estimator_window,
                "leading_eigenvectors": leading_eigenvectors,
                "variogram_lag_max": int(min(variogram_lag_max, max(0, len(retained_pivot) - 1))),
                "num_requested_dates": int(len(target_dates)),
                "num_dates": int(len(summary_frame)),
                "first_date": str(summary_frame.iloc[0]["date"]),
                "last_date": str(summary_frame.iloc[-1]["date"]),
                "num_assets": int(num_assets),
                "skipped_dates": skipped_dates,
                "quality_report": quality_report,
            },
        )
        files = write_quality_artifacts(outdir, quality_report)
        files.update({
            "summary_frame": summary_path,
            "spectrum_frame": spectrum_path,
            "eigenvalue_variogram": variogram_path,
            "eigenvector_similarity": alignment_path,
        })
        if request_path is not None:
            files["request"] = request_path
        if summary_json_path is not None:
            files["summary"] = summary_json_path

    return InspectionIntervalResult(
        request=request,
        universe=universe.name,
        cleaning_method=cleaning_method,
        input_type=input_type,
        matrix_type=matrix_type,
        estimator_method=estimator_method,
        estimator_window=estimator_window,
        observation_dates=tuple(pd.Timestamp(date) for date in pd.to_datetime(summary_frame["date"])),
        num_assets=num_assets,
        summary_frame=summary_frame,
        spectrum_frame=spectrum_frame,
        variogram_frame=variogram_frame,
        eigenvector_similarity_frame=eigenvector_similarity_frame,
        quality_report=quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )
