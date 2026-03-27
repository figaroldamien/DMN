from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Keep matplotlib cache in a writable temp directory in this environment.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "optimal_tf_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .features import compute_returns, sanitize_returns


def equal_weight_rebalanced_benchmark(prices: pd.DataFrame, *, max_abs_return: float | None = None) -> pd.Series:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=max_abs_return)
    return returns.mean(axis=1).fillna(0.0)


def equal_weight_buy_and_hold_benchmark(prices: pd.DataFrame, *, max_abs_return: float | None = None) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0 / prices.shape[1], index=prices.columns, dtype=float)
    returns = sanitize_returns(compute_returns(prices), max_abs_return=max_abs_return).fillna(0.0)
    return (returns * weights).sum(axis=1)


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
