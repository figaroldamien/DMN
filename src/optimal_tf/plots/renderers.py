from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "optimal_tf_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def cumulative_nav(returns: pd.Series, base: float = 1.0) -> pd.Series:
    return base * (1.0 + returns.fillna(0.0)).cumprod()


def render_evaluation_plot(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    buy_hold_returns: pd.Series,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    nav = cumulative_nav(portfolio_returns)
    benchmark_nav = cumulative_nav(benchmark_returns).reindex(nav.index).ffill()
    buy_hold_nav = cumulative_nav(buy_hold_returns).reindex(nav.index).ffill()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(nav.index, nav.values, label="optimal_tf portfolio", linewidth=2.2)
    ax.plot(benchmark_nav.index, benchmark_nav.values, label="universe equal-weight index", linewidth=1.8)
    ax.plot(buy_hold_nav.index, buy_hold_nav.values, label="equal-weight buy and hold", linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def render_series_comparison_plot(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    ylabel: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for column in frame.columns:
        ax.plot(frame.index, frame[column], label=str(column), linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
