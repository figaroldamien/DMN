"""Freshness policy helpers for cached market data."""

from __future__ import annotations

import pandas as pd

from .models import TickerCacheMetadata


SUPPORTED_REFRESH_POLICIES = ('auto', 'always', 'never')


def validate_refresh_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized not in SUPPORTED_REFRESH_POLICIES:
        raise ValueError(
            f'Unsupported refresh policy {policy!r}. '
            f'Allowed values: {list(SUPPORTED_REFRESH_POLICIES)}'
        )
    return normalized


def should_refresh_ticker(
    metadata: TickerCacheMetadata | None,
    *,
    refresh_policy: str,
    now: pd.Timestamp | None = None,
) -> bool:
    policy = validate_refresh_policy(refresh_policy)
    if policy == 'always':
        return True
    if metadata is None:
        return policy != 'never'
    if policy == 'never':
        return False
    if metadata.data_end is None:
        return True
    now_ts = pd.Timestamp.now(tz='UTC').tz_localize(None) if now is None else pd.Timestamp(now).tz_localize(None) if pd.Timestamp(now).tzinfo is not None else pd.Timestamp(now)
    data_end = pd.Timestamp(metadata.data_end)
    # Simple and intentionally conservative freshness rule for the MVP:
    # if we already have data through yesterday (or later), keep the cache.
    expected_min = (now_ts.normalize() - pd.Timedelta(days=1))
    return data_end.normalize() < expected_min
