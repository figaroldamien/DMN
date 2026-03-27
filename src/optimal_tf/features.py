from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    if log_returns:
        return np.log(prices).diff()
    return prices.pct_change()


def sanitize_returns(
    returns: pd.DataFrame | pd.Series,
    *,
    max_abs_return: float | None = None,
) -> pd.DataFrame | pd.Series:
    if max_abs_return is None:
        return returns
    cleaned = returns.copy()
    cleaned = cleaned.where(cleaned.abs() <= max_abs_return)
    return cleaned


def ewma_vol(returns: pd.DataFrame, span: int = 60, min_periods: int | None = None) -> pd.DataFrame:
    min_periods = span if min_periods is None else min_periods
    return returns.ewm(span=span, adjust=False, min_periods=min_periods).std()


def normalize_returns_by_vol(returns: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    return returns / (vol + 1e-12)


def trend_ema_signal(returns: pd.DataFrame, span: int = 63) -> pd.DataFrame:
    return returns.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_corr_frame(frame: pd.DataFrame, window: int, min_periods: int | None = None) -> dict[pd.Timestamp, tuple[pd.DataFrame, int]]:
    min_periods = window if min_periods is None else min_periods
    out: dict[pd.Timestamp, tuple[pd.DataFrame, int]] = {}
    for idx in range(len(frame)):
        end = frame.index[idx]
        sample = frame.iloc[max(0, idx - window + 1) : idx + 1]
        sample = sample.dropna(how="any")
        if len(sample) < min_periods:
            continue
        out[end] = (sample.corr(), len(sample))
    return out
