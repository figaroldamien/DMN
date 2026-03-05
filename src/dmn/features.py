from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(returns: pd.DataFrame, span: int = 60, min_periods: int = 60) -> pd.DataFrame:
    return returns.ewm(span=span, adjust=False, min_periods=min_periods).std()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def rolling_return(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.pct_change(window)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(prices: pd.DataFrame, short_span: int, long_span: int) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for c in prices.columns:
        out[c] = ema(prices[c], short_span) - ema(prices[c], long_span)
    return out


def phi(y: pd.DataFrame) -> pd.DataFrame:
    return (y * np.exp(-(y ** 2) / 4.0)) / 0.89


def make_dmn_features(prices: pd.DataFrame, daily_vol: pd.DataFrame) -> pd.DataFrame:
    feats = []

    horizons = [1, 21, 63, 126, 252]
    for h in horizons:
        rr = rolling_return(prices, h)
        norm = rr / (daily_vol * np.sqrt(h))
        norm.columns = pd.MultiIndex.from_product([[f"ret_{h}"], norm.columns])
        feats.append(norm)

    price_std63 = prices.rolling(63).std()
    for s, l in [(8, 24), (16, 48), (32, 96)]:
        m = macd(prices, s, l)
        q = m / (price_std63 + 1e-12)
        z_std252 = q.rolling(252).std()
        y = q / (z_std252 + 1e-12)
        y.columns = pd.MultiIndex.from_product([[f"macdY_{s}_{l}"], y.columns])
        feats.append(y)

    return pd.concat(feats, axis=1).sort_index(axis=1)
