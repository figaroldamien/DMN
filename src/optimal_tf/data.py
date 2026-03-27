from __future__ import annotations

from typing import Iterable

import pandas as pd

from market_tickers_data.universes import MARKET_TICKERS

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def tickers_for_universe(name: str) -> list[str]:
    if name not in MARKET_TICKERS:
        raise KeyError(f"Unknown universe '{name}'. Allowed values: {sorted(MARKET_TICKERS)}")
    return list(MARKET_TICKERS[name])


def load_prices_yf(tickers: Iterable[str], start: str = "2000-01-01") -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance not installed. pip install yfinance")
    tickers = list(tickers)
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data.rename(columns={"Close": tickers[0]})[[tickers[0]]]
    return prices.dropna(how="all").ffill()


def load_prices_for_universe(universe: str, start: str = "2000-01-01") -> pd.DataFrame:
    return load_prices_yf(tickers_for_universe(universe), start=start)
