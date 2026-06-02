from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from optimal_tf.config import BacktestConfig, EstimationConfig, EvaluationConfig
from optimal_tf.evaluation import evaluate_portfolio
from trading_core.features import compute_returns, ewma_vol, normalize_returns_by_vol, sanitize_returns
from trading_core.reporting import cumulative_nav
from trading_core.reporting.plots import plt
from trading_core.risk import clean_correlation_matrix
from trading_core.risk.pipeline import _resolve_covariance_window, rolling_corr_frame


def build_normalized_returns(prices: pd.DataFrame, estimation: EstimationConfig) -> pd.DataFrame:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=estimation.max_abs_return)
    vol = ewma_vol(returns, span=estimation.vol_span)
    return normalize_returns_by_vol(returns, vol)


def matrix_sample(
    prices: pd.DataFrame,
    estimation: EstimationConfig,
    matrix_date: pd.Timestamp,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    z_returns = build_normalized_returns(prices, estimation)
    raw_corr = rolling_corr_frame(
        z_returns,
        window=_resolve_covariance_window(estimation),
        min_periods=estimation.covariance_min_periods,
        target_dates=pd.DatetimeIndex([matrix_date]),
    )
    if matrix_date not in raw_corr:
        raise ValueError(
            f"No correlation sample available on {matrix_date.date()} for covariance_window={estimation.covariance_window}."
        )
    return raw_corr[matrix_date]


def matrix_benchmark_rows(
    prices: pd.DataFrame,
    estimation: EstimationConfig,
    methods: list[str],
    matrix_date: pd.Timestamp,
) -> list[dict[str, float | int | str]]:
    empirical_corr, sample_size, sample_frame = matrix_sample(prices, estimation, matrix_date)
    complete_rows = int(sample_frame.dropna(axis=0, how="any").shape[0])
    rows: list[dict[str, float | int | str]] = []
    empirical_arr = empirical_corr.to_numpy(dtype=float)
    empirical_offdiag = empirical_arr[~np.eye(len(empirical_arr), dtype=bool)]
    empirical_fro = float(np.linalg.norm(empirical_arr, ord="fro"))

    for method in methods:
        cleaned = clean_correlation_matrix(
            empirical_corr,
            data=sample_frame,
            sample_size=sample_size,
            method=method,
            linear_shrinkage=estimation.linear_shrinkage,
            bandwidth=estimation.rie_bandwidth,
        )
        arr = cleaned.to_numpy(dtype=float)
        offdiag = arr[~np.eye(len(arr), dtype=bool)]
        diff = arr - empirical_arr
        eigvals = np.linalg.eigvalsh(arr)
        rows.append(
            {
                "date": matrix_date.strftime("%Y-%m-%d"),
                "method": method,
                "num_assets": int(len(cleaned)),
                "sample_size": int(sample_size),
                "complete_case_rows": complete_rows,
                "mean_abs_offdiag": float(np.mean(np.abs(offdiag))) if len(offdiag) else 0.0,
                "max_abs_offdiag": float(np.max(np.abs(offdiag))) if len(offdiag) else 0.0,
                "fro_norm": float(np.linalg.norm(arr, ord="fro")),
                "fro_vs_empirical": float(np.linalg.norm(diff, ord="fro")),
                "fro_vs_empirical_pct": float(np.linalg.norm(diff, ord="fro") / max(empirical_fro, 1e-12)),
                "eig_min": float(np.min(eigvals)),
                "eig_max": float(np.max(eigvals)),
                "condition_number": float(np.max(eigvals) / max(np.min(eigvals), 1e-12)),
                "mean_diag": float(np.mean(np.diag(arr))),
                "mean_abs_empirical_offdiag": float(np.mean(np.abs(empirical_offdiag))) if len(empirical_offdiag) else 0.0,
            }
        )
    return rows


def strategy_benchmark_rows(
    prices: pd.DataFrame,
    estimation: EstimationConfig,
    backtest: BacktestConfig,
    evaluation: EvaluationConfig,
    methods: list[str],
    strategies: list[str],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        est_cfg = replace(estimation, cleaning_method=method)
        for strategy in strategies:
            eval_cfg = replace(evaluation, strategy=strategy)
            result = evaluate_portfolio(prices, est_cfg, backtest, eval_cfg)
            payload = asdict(result.summary)
            payload.update(
                {
                    "method": method,
                    "strategy": strategy,
                    "final_nav": float(cumulative_nav(result.daily_returns_net).iloc[-1]) if len(result.daily_returns_net) else 1.0,
                }
            )
            rows.append(payload)
    return rows


def reference_pipe_row(
    prices: pd.DataFrame,
    estimation: EstimationConfig,
    matrix_date: pd.Timestamp,
) -> dict[str, float | int | str] | None:
    try:
        import rie_estimator
    except ModuleNotFoundError:
        return None

    returns = sanitize_returns(compute_returns(prices), max_abs_return=estimation.max_abs_return)
    window = _resolve_covariance_window(estimation)
    pos = returns.index.get_indexer([matrix_date])[0]
    if pos < 0:
        return None
    sample = returns.iloc[max(0, pos - window + 1) : pos + 1].dropna(axis=0, how="any")
    if sample.empty or sample.shape[1] < 2:
        return None

    cleaned = rie_estimator.get_rie(sample.astype(float), normalize=True, max_ones=True)
    arr = np.asarray(cleaned, dtype=float)
    diag = np.sqrt(np.clip(np.diag(arr), 1e-12, None))
    arr = arr / np.outer(diag, diag)
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 1.0)
    eigvals = np.linalg.eigvalsh(arr)
    offdiag = arr[~np.eye(len(arr), dtype=bool)]
    return {
        "date": matrix_date.strftime("%Y-%m-%d"),
        "method": "rie_reference_pipe",
        "num_assets": int(sample.shape[1]),
        "sample_size": int(sample.shape[0]),
        "complete_case_rows": int(sample.shape[0]),
        "mean_abs_offdiag": float(np.mean(np.abs(offdiag))) if len(offdiag) else 0.0,
        "max_abs_offdiag": float(np.max(np.abs(offdiag))) if len(offdiag) else 0.0,
        "fro_norm": float(np.linalg.norm(arr, ord="fro")),
        "fro_vs_empirical": float("nan"),
        "fro_vs_empirical_pct": float("nan"),
        "eig_min": float(np.min(eigvals)),
        "eig_max": float(np.max(eigvals)),
        "condition_number": float(np.max(eigvals) / max(np.min(eigvals), 1e-12)),
        "mean_diag": float(np.mean(np.diag(arr))),
        "mean_abs_empirical_offdiag": float("nan"),
    }


def eigenvalue_rows(
    empirical_corr: pd.DataFrame,
    sample_size: int,
    sample_frame: pd.DataFrame,
    estimation: EstimationConfig,
    methods: list[str],
    *,
    matrix_date: pd.Timestamp,
) -> list[dict[str, float | int | str]]:
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
                    "covariance_window": int(estimation.covariance_window or 0),
                    "covariance_min_periods": int(estimation.covariance_min_periods),
                    "method": method,
                    "rank": rank,
                    "eigenvalue": float(eigenvalue),
                    "variance_share": float(eigenvalue / total) if total else 0.0,
                    "cumulative_variance_share": float(cumulative[rank - 1] / total) if total else 0.0,
                }
            )
    return rows


def render_scree_overview(frame: pd.DataFrame, output_path: Path, *, log_scale: bool) -> Path:
    windows = sorted(frame["covariance_window"].unique())
    n_panels = len(windows)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 4.5 * n_rows), sharex=True)
    axes_array = np.atleast_1d(axes).reshape(n_rows, n_cols)
    flat_axes = axes_array.flatten()

    for ax, window in zip(flat_axes, windows):
        panel = frame.loc[frame["covariance_window"] == window]
        pivot = panel.pivot(index="rank", columns="method", values="eigenvalue")
        for method in pivot.columns:
            ax.plot(pivot.index, pivot[method], label=str(method), linewidth=1.9)
        min_periods = int(panel["covariance_min_periods"].iloc[0])
        ax.set_title(f"window={window}, min_periods={min_periods}")
        ax.set_xlabel("Eigenvalue rank")
        ax.set_ylabel("Eigenvalue")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("Cleaner scree plot by covariance window")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_strategy_pivots(frame: pd.DataFrame, outdir: Path) -> None:
    for metric in ("total_return", "sharpe", "mdd", "avg_turnover"):
        pivot = frame.pivot_table(index="covariance_window", columns=["method", "strategy"], values=metric)
        pivot.sort_index().to_csv(outdir / f"strategy_{metric}_pivot.csv")


def write_matrix_pivots(frame: pd.DataFrame, outdir: Path) -> None:
    for metric in ("condition_number", "eig_max", "eig_min", "mean_abs_offdiag", "fro_vs_empirical_pct"):
        pivot = frame.pivot(index="covariance_window", columns="method", values=metric)
        pivot.sort_index().to_csv(outdir / f"matrix_{metric}_pivot.csv")
