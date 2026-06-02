"""Price-loading entry points for market_tickers_data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from market_tickers_data.universes import MARKET_TICKERS

from .cache import (
    ensure_cache_dirs,
    read_cached_ticker_prices,
    read_ticker_metadata,
    write_cached_ticker_prices,
    write_ticker_metadata,
)
from .freshness import should_refresh_ticker, validate_refresh_policy
from .models import TickerCacheMetadata

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


_YF_TICKER_MAP = {
    'BRK.B': 'BRK-B',
    'BF.B': 'BF-B',
}


def _map_ticker_for_yfinance(ticker: str) -> str:
    return _YF_TICKER_MAP.get(ticker, ticker)


def _download_prices_yf(tickers: Iterable[str], start: str = '2000-01-01') -> pd.DataFrame:
    if yf is None:
        raise ImportError('yfinance not installed. pip install yfinance')
    requested = list(tickers)
    if not requested:
        return pd.DataFrame()
    resolved = [_map_ticker_for_yfinance(ticker) for ticker in requested]
    data = yf.download(resolved, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'].copy()
    else:
        prices = data.rename(columns={'Close': resolved[0]})[[resolved[0]]]
    prices = prices.dropna(how='all').ffill()
    rename_back = {mapped: original for original, mapped in zip(requested, resolved)}
    return prices.rename(columns=rename_back)


def _load_single_ticker(
    ticker: str,
    *,
    start: str,
    refresh_policy: str,
    now: pd.Timestamp | None = None,
) -> pd.Series:
    metadata = read_ticker_metadata(ticker)
    cached = read_cached_ticker_prices(ticker)
    refresh = should_refresh_ticker(metadata, refresh_policy=refresh_policy, now=now)
    if not refresh and cached is not None:
        return cached.loc[cached.index >= pd.Timestamp(start)].copy()
    if refresh_policy == 'never' and cached is None:
        raise FileNotFoundError(f'No cached prices available for ticker {ticker!r}.')
    downloaded = _download_prices_yf([ticker], start=start)
    if ticker not in downloaded.columns:
        raise ValueError(f'No downloaded prices returned for ticker {ticker!r}.')
    series = downloaded[ticker].dropna().copy()
    write_cached_ticker_prices(ticker, series)
    write_ticker_metadata(
        TickerCacheMetadata(
            ticker=ticker,
            provider='yahoo',
            last_fetch_at=pd.Timestamp.now(tz='UTC').isoformat(),
            data_start=series.index.min().strftime('%Y-%m-%d') if not series.empty else None,
            data_end=series.index.max().strftime('%Y-%m-%d') if not series.empty else None,
            rows=int(len(series)),
        )
    )
    return series.loc[series.index >= pd.Timestamp(start)].copy()


def load_prices_yf(
    tickers: Iterable[str],
    start: str = '2000-01-01',
    *,
    refresh_policy: str = 'auto',
) -> pd.DataFrame:
    policy = validate_refresh_policy(refresh_policy)
    ensure_cache_dirs()
    requested = list(tickers)
    series_map: dict[str, pd.Series] = {}
    for ticker in requested:
        series_map[ticker] = _load_single_ticker(ticker, start=start, refresh_policy=policy)
    if not series_map:
        return pd.DataFrame()
    frame = pd.concat(series_map, axis=1).sort_index().ffill()
    frame.columns = list(series_map.keys())
    return frame.dropna(how='all')


def load_prices_for_universe(
    universe: str,
    start: str = '2000-01-01',
    *,
    refresh_policy: str = 'auto',
) -> pd.DataFrame:
    if universe not in MARKET_TICKERS:
        raise KeyError(f"Unknown universe '{universe}'. Allowed values: {sorted(MARKET_TICKERS)}")
    return load_prices_yf(MARKET_TICKERS[universe], start=start, refresh_policy=refresh_policy)
