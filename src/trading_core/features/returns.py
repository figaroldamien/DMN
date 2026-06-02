from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    if log_returns:
        return np.log(prices).diff()
    return prices.pct_change()


def rolling_return(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.pct_change(window)


def sanitize_returns(
    returns: pd.DataFrame | pd.Series,
    *,
    max_abs_return: float | None = None,
) -> pd.DataFrame | pd.Series:
    if max_abs_return is None:
        return returns
    cleaned = returns.copy()
    return cleaned.where(cleaned.abs() <= max_abs_return)

