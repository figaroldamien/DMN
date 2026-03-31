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


def alpha_from_span(span: int | None) -> float | None:
    if span is None:
        return None
    if span <= 0:
        raise ValueError("span must be strictly positive.")
    return 2.0 / (float(span) + 1.0)


def effective_span_from_alpha(alpha: float | None) -> int | None:
    if alpha is None:
        return None
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    return max(1, int(round((2.0 / alpha) - 1.0)))


def resolve_ewma_alpha(
    *,
    alpha: float | None = None,
    span: int | None = None,
) -> float:
    if alpha is not None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        return float(alpha)
    resolved = alpha_from_span(span)
    if resolved is None:
        raise ValueError("Either alpha or span must be provided.")
    return resolved


def ewma_vol(
    returns: pd.DataFrame,
    *,
    alpha: float | None = None,
    span: int | None = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    alpha = resolve_ewma_alpha(alpha=alpha, span=span)
    effective_span = effective_span_from_alpha(alpha)
    min_periods = effective_span if min_periods is None else min_periods
    return returns.ewm(alpha=alpha, adjust=False, min_periods=min_periods).std()


def normalize_returns_by_vol(returns: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    return returns / (vol + 1e-12)


def trend_ema_signal(
    returns: pd.DataFrame,
    *,
    alpha: float | None = None,
    span: int | None = 63,
    min_periods: int | None = None,
) -> pd.DataFrame:
    alpha = resolve_ewma_alpha(alpha=alpha, span=span)
    effective_span = effective_span_from_alpha(alpha)
    min_periods = effective_span if min_periods is None else min_periods
    return returns.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()


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


def ewma_cov_frame(
    frame: pd.DataFrame,
    *,
    alpha: float,
    min_periods: int,
) -> dict[pd.Timestamp, tuple[pd.DataFrame, int]]:
    alpha = resolve_ewma_alpha(alpha=alpha)
    cov = frame.ewm(alpha=alpha, adjust=False, min_periods=min_periods).cov()
    effective_span = effective_span_from_alpha(alpha)
    out: dict[pd.Timestamp, tuple[pd.DataFrame, int]] = {}
    available_dates = set(cov.index.get_level_values(0))
    for ts in frame.index:
        if ts not in available_dates:
            continue
        matrix = cov.loc[ts]
        matrix = matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if matrix.empty:
            continue
        tickers = [ticker for ticker in matrix.index if ticker in matrix.columns]
        matrix = matrix.loc[tickers, tickers]
        if matrix.empty:
            continue
        seen = int(frame.loc[:ts].dropna(how="all").shape[0])
        out[ts] = (matrix.astype(float), min(seen, effective_span or seen))
    return out
