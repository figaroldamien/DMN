"""Typed models for market price loading and cache metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceLoadRequest:
    target: str
    start: str = '2000-01-01'
    refresh_policy: str = 'auto'


@dataclass(frozen=True)
class TickerCacheMetadata:
    ticker: str
    provider: str
    last_fetch_at: str
    data_start: str | None
    data_end: str | None
    rows: int
