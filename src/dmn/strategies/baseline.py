from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import macd, phi, rolling_return


def strategy_long_only(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


def strategy_sgn_12m(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    rr = rolling_return(prices, lookback)
    return np.sign(rr).clip(-1.0, 1.0)


def strategy_baz_macd(prices: pd.DataFrame) -> pd.DataFrame:
    price_std63 = prices.rolling(63).std()
    ys = []
    for s, l in [(8, 24), (16, 48), (32, 96)]:
        m = macd(prices, s, l)
        q = m / (price_std63 + 1e-12)
        z_std252 = q.rolling(252).std()
        y = q / (z_std252 + 1e-12)
        ys.append(y)
    ybar = sum(ys) / len(ys)
    return phi(ybar).clip(-1.0, 1.0)
