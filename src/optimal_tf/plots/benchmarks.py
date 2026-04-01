from __future__ import annotations

import pandas as pd

from ..features import compute_returns, sanitize_returns


def equal_weight_rebalanced_benchmark(prices: pd.DataFrame, *, max_abs_return: float | None = None) -> pd.Series:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=max_abs_return)
    return returns.mean(axis=1).fillna(0.0)


def equal_weight_buy_and_hold_benchmark(prices: pd.DataFrame, *, max_abs_return: float | None = None) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0 / prices.shape[1], index=prices.columns, dtype=float)
    returns = sanitize_returns(compute_returns(prices), max_abs_return=max_abs_return).fillna(0.0)
    return (returns * weights).sum(axis=1)
