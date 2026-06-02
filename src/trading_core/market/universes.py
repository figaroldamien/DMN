from __future__ import annotations

from typing import Any

from market_tickers_data.universes import MARKET_BENCHMARKS, MARKET_TICKERS, MARKET_UNIVERSE_METADATA


def list_universes() -> list[str]:
    return sorted(MARKET_TICKERS)


def get_universe_tickers(name: str) -> list[str]:
    if name not in MARKET_TICKERS:
        raise KeyError(f"Unknown universe '{name}'. Allowed values: {list_universes()}")
    return list(MARKET_TICKERS[name])


def get_universe_metadata(name: str) -> dict[str, Any]:
    if name not in MARKET_UNIVERSE_METADATA:
        raise KeyError(f"Unknown universe '{name}'. Allowed values: {list_universes()}")
    return dict(MARKET_UNIVERSE_METADATA[name])


def get_universe_benchmark(name: str) -> dict[str, Any] | None:
    benchmark = MARKET_BENCHMARKS.get(name)
    return None if benchmark is None else dict(benchmark)
