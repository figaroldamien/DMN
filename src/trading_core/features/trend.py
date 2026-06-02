from __future__ import annotations

import pandas as pd

from .volatility import effective_span_from_alpha, resolve_ewma_alpha


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(prices: pd.DataFrame, short_span: int, long_span: int) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for column in prices.columns:
        out[column] = ema(prices[column], short_span) - ema(prices[column], long_span)
    return out


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

