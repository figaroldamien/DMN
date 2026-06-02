from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_tickers_data.components import (
    CAC40_COMPONENTS,
    DATASET_COMPONENTS,
    DJI_COMPONENTS,
    EUROSTOXX50_COMPONENTS,
    INDEX_COMPONENTS,
    NASDAQ100_COMPONENTS,
    SP500_COMPONENTS,
    WORLD_INDEX_COMPONENTS,
)
from optimal_tf.allocation import compute_strategy_state_at_date, supported_strategies
from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.features import trend_ema_signal
from optimal_tf.scripts.common import (
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
from trading_core.reporting.plots import plt

from optimal_tf.strategies.common import resolve_allocation_date, resolve_covariance_cache_until_date, sanitized_normalized_returns

from .io import ensure_output_dir, write_json, write_request_json
from .models import (
    EigenvectorInspectionRequest,
    EigenvectorInspectionResult,
    InspectionSnapshotRequest,
    InspectionSnapshotResult,
    RunArtifacts,
)

DEFAULT_WINDOWS = (40, 60, 80, 120, 252, 504, 1200)
UNIVERSE_COMPONENTS = {
    "nasdaq100": NASDAQ100_COMPONENTS,
    "cac40": CAC40_COMPONENTS,
    "dji": DJI_COMPONENTS,
    "eurostoxx50": EUROSTOXX50_COMPONENTS,
    "sp500": SP500_COMPONENTS,
    "index": INDEX_COMPONENTS,
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


def _sector_metadata(universe: str, tickers: list[str]) -> pd.DataFrame:
    components = UNIVERSE_COMPONENTS.get(universe, {})
    rows: list[dict[str, str]] = []
    for ticker in tickers:
        meta = components.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "sector": str(meta.get("sector", "zzz_unknown")).strip() or "zzz_unknown",
                "sub_sector": str(meta.get("sub_sector", "zzz_unknown")).strip() or "zzz_unknown",
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
    windows = request.windows or list(DEFAULT_WINDOWS)
    windows = parse_windows(",".join(str(item) for item in windows))
    if request.selection_top_n <= 0:
        raise ValueError("selection_top_n must be strictly positive.")
    if request.selection_cumulative_variance <= 0.0 or request.selection_cumulative_variance > 100.0:
        raise ValueError("selection_cumulative_variance must be in (0, 100].")
    if request.method not in supported_cleaning_methods():
        raise ValueError(f"Unknown cleaning method {request.method!r}.")

    prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=request.refresh_policy)
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
        }
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        req = write_request_json(outdir, request)
        files = {
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
        }
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
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )


def _sorted_metadata(universe: str, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    metadata = _sector_metadata(universe, tickers)
    sorted_tickers = list(metadata.index)
    return metadata, sorted_tickers


def _sorted_matrix(frame: pd.DataFrame, sorted_tickers: list[str]) -> pd.DataFrame:
    tickers = [ticker for ticker in sorted_tickers if ticker in frame.index and ticker in frame.columns]
    return frame.loc[tickers, tickers]


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

    strategy = request.strategy or evaluation.strategy
    if strategy not in supported_strategies():
        raise ValueError(f"Unknown strategy {strategy!r}.")

    cleaning_method = request.cleaning_method or estimation.cleaning_method
    if cleaning_method not in supported_cleaning_methods():
        raise ValueError(f"Unknown cleaning method {cleaning_method!r}.")

    covariance_window = int(request.covariance_window or estimation.covariance_window or 252)
    snapshot_estimation = resolve_window_estimation_cfg(
        estimation,
        covariance_window,
        min_periods_mode="clamp",
    )
    snapshot_estimation = snapshot_estimation.__class__(**{
        **snapshot_estimation.__dict__,
        "cleaning_method": cleaning_method,
    })
    long_only = bool(backtest.long_only if request.long_only is None else request.long_only)

    prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=request.refresh_policy)
    allocation_date = resolve_allocation_date(prices.index, as_of_date=request.date)
    history = prices.loc[prices.index <= allocation_date]
    if history.empty:
        raise ValueError(f"No price history available on or before {allocation_date.date()}.")

    empirical_corr, sample_size, sample_frame = matrix_sample(history, snapshot_estimation, allocation_date)
    cleaned_corr = clean_correlation_matrix(
        empirical_corr,
        data=sample_frame,
        sample_size=sample_size,
        method=cleaning_method,
        linear_shrinkage=snapshot_estimation.linear_shrinkage,
        bandwidth=snapshot_estimation.rie_bandwidth,
    )

    returns, vol, z_returns = sanitized_normalized_returns(history, snapshot_estimation)
    trend = trend_ema_signal(z_returns, alpha=snapshot_estimation.trend_alpha, span=snapshot_estimation.trend_span)
    vol_t = vol.loc[:allocation_date].ffill().iloc[-1].reindex(cleaned_corr.index).fillna(0.0)
    cleaned_cov = correlation_to_covariance(cleaned_corr, vol_t)

    covariance_cache = resolve_covariance_cache_until_date(history, snapshot_estimation, allocation_date)
    state = compute_strategy_state_at_date(
        history,
        snapshot_estimation,
        strategy,
        date=allocation_date,
        long_only=long_only,
        covariance_cache=covariance_cache,
    )

    metadata, sorted_tickers = _sorted_metadata(universe.name, list(cleaned_corr.index))
    sorted_sample_corr = _sorted_matrix(empirical_corr, sorted_tickers)
    sorted_cleaned_corr = _sorted_matrix(cleaned_corr, sorted_tickers)
    sorted_cleaned_cov = _sorted_matrix(cleaned_cov, sorted_tickers)

    corr_spectrum, corr_eigenvectors = _eigenvector_frame(sorted_cleaned_corr, universe=universe.name, prefix="corr_ev")
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

    allocation_frame = pd.DataFrame(
        {
            "base_weight": state.base_weights.reindex(sorted_tickers).fillna(0.0).to_numpy(dtype=float),
            "effective_weight": state.effective_weights.reindex(sorted_tickers).fillna(0.0).to_numpy(dtype=float),
        },
        index=feature_index,
    )
    allocation_frame["abs_effective_weight"] = allocation_frame["effective_weight"].abs()

    outdir = ensure_output_dir(request.output_dir or "output/optimal_tf/inspection/snapshot")
    files: dict[str, Path] = {}
    if outdir is not None:
        sample_corr_path = outdir / "sample_correlation.csv"
        cleaned_corr_path = outdir / "cleaned_correlation.csv"
        cleaned_cov_path = outdir / "cleaned_covariance.csv"
        corr_spectrum_path = outdir / "correlation_spectrum.csv"
        cov_spectrum_path = outdir / "covariance_spectrum.csv"
        corr_vectors_path = outdir / "correlation_eigenvectors.csv"
        cov_vectors_path = outdir / "covariance_eigenvectors.csv"
        features_path = outdir / "features.csv"
        allocation_path = outdir / "allocation.csv"
        sorted_sample_corr.to_csv(sample_corr_path)
        sorted_cleaned_corr.to_csv(cleaned_corr_path)
        sorted_cleaned_cov.to_csv(cleaned_cov_path)
        corr_spectrum.to_csv(corr_spectrum_path, index=False)
        cov_spectrum.to_csv(cov_spectrum_path, index=False)
        corr_eigenvectors.to_csv(corr_vectors_path)
        cov_eigenvectors.to_csv(cov_vectors_path)
        feature_frame.to_csv(features_path)
        allocation_frame.to_csv(allocation_path)
        request_path = write_request_json(outdir, request)
        summary_path = write_json(
            outdir,
            "summary.json",
            {
                "universe": universe.name,
                "strategy": strategy,
                "cleaning_method": cleaning_method,
                "covariance_window": covariance_window,
                "allocation_date": allocation_date.strftime("%Y-%m-%d"),
                "num_assets": int(len(sorted_tickers)),
                "sample_size": int(sample_size),
                "signal_scale": float(state.signal_scale),
                "long_only": bool(long_only),
            },
        )
        files = {
            "sample_correlation": sample_corr_path,
            "cleaned_correlation": cleaned_corr_path,
            "cleaned_covariance": cleaned_cov_path,
            "correlation_spectrum": corr_spectrum_path,
            "covariance_spectrum": cov_spectrum_path,
            "correlation_eigenvectors": corr_vectors_path,
            "covariance_eigenvectors": cov_vectors_path,
            "features": features_path,
            "allocation": allocation_path,
        }
        if request_path is not None:
            files["request"] = request_path
        if summary_path is not None:
            files["summary"] = summary_path

    return InspectionSnapshotResult(
        request=request,
        universe=universe.name,
        strategy=strategy,
        cleaning_method=cleaning_method,
        covariance_window=covariance_window,
        allocation_date=allocation_date,
        sample_size=int(sample_size),
        num_assets=int(len(sorted_tickers)),
        signal_scale=float(state.signal_scale),
        sample_correlation=sorted_sample_corr,
        cleaned_correlation=sorted_cleaned_corr,
        cleaned_covariance=sorted_cleaned_cov,
        correlation_spectrum=corr_spectrum,
        covariance_spectrum=cov_spectrum,
        correlation_eigenvectors=corr_eigenvectors,
        covariance_eigenvectors=cov_eigenvectors,
        feature_frame=feature_frame,
        allocation_frame=allocation_frame,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )
