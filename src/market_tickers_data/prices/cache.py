"""Disk-cache helpers for market price data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .models import TickerCacheMetadata


CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "market_prices_cache"
YAHOO_CACHE_ROOT = CACHE_ROOT / "yahoo"
YAHOO_TICKER_CACHE_DIR = YAHOO_CACHE_ROOT / "tickers"
YAHOO_METADATA_CACHE_DIR = YAHOO_CACHE_ROOT / "metadata"


def ensure_cache_dirs() -> None:
    YAHOO_TICKER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    YAHOO_METADATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ticker_cache_path(ticker: str) -> Path:
    ensure_cache_dirs()
    return YAHOO_TICKER_CACHE_DIR / f"{ticker}.parquet"


def ticker_metadata_path(ticker: str) -> Path:
    ensure_cache_dirs()
    return YAHOO_METADATA_CACHE_DIR / f"{ticker}.json"


def read_cached_ticker_prices(ticker: str) -> pd.Series | None:
    path = ticker_cache_path(ticker)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.shape[1] == 0:
        return None
    data = frame.iloc[:, 0].copy()
    data.index = pd.to_datetime(data.index)
    data.name = ticker
    return data.sort_index()


def write_cached_ticker_prices(ticker: str, prices: pd.Series) -> None:
    path = ticker_cache_path(ticker)
    series = prices.copy()
    series.name = ticker
    series.sort_index().to_frame(name=ticker).to_parquet(path, engine="pyarrow")


def read_ticker_metadata(ticker: str) -> TickerCacheMetadata | None:
    path = ticker_metadata_path(ticker)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return TickerCacheMetadata(**payload)


def write_ticker_metadata(metadata: TickerCacheMetadata) -> None:
    path = ticker_metadata_path(metadata.ticker)
    path.write_text(json.dumps(asdict(metadata), indent=2) + "\n")
