from __future__ import annotations

import pandas as pd


_FREQ_MAP = {
    "daily": "D",
    "weekly": "W-FRI",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}


def supported_rebalance_frequencies() -> list[str]:
    return sorted(_FREQ_MAP)


def resolve_rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: str,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    if len(index) == 0:
        return pd.DatetimeIndex([])
    if frequency not in _FREQ_MAP:
        raise ValueError(f"Unknown rebalance frequency '{frequency}'. Allowed values: {sorted(_FREQ_MAP)}")

    dates = pd.DatetimeIndex(index).sort_values().unique()
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    if frequency == "daily":
        return dates

    marker = pd.Series(dates, index=dates)
    grouped = marker.resample(_FREQ_MAP[frequency]).last().dropna()
    return pd.DatetimeIndex(grouped.to_numpy())
