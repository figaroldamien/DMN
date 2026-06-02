"""Named universe catalogs built from the raw universe component definitions."""

from __future__ import annotations

from typing import Any, Dict, List

from .load import (
    CAC40,
    DATASET,
    DATASET_ALL,
    DJI,
    EUROSTOXX50,
    INDEX,
    NASDAQ100,
    SP500,
    WORLD_INDEX,
    get_universe_benchmark,
    get_universe_metadata,
)

TABLE8_ASSETS_BY_CATEGORY: Dict[str, List[str]] = DATASET
TABLE8_ALL: List[str] = DATASET_ALL

TEST: List[str] = ['^FCHI']
MARKET_TICKERS: Dict[str, List[str]] = {
    'nasdaq100': NASDAQ100,
    'cac40': CAC40,
    'dji': DJI,
    'eurostoxx50': EUROSTOXX50,
    'sp500': SP500,
    'index': INDEX,
    'world_index': WORLD_INDEX,
    'dataset_all': DATASET_ALL,
    'table8_all': DATASET_ALL,
    'test': TEST,
}
MARKET_UNIVERSE_METADATA: Dict[str, dict[str, Any]] = {
    name: get_universe_metadata(name)
    for name in ['nasdaq100', 'cac40', 'dji', 'eurostoxx50', 'sp500', 'index', 'world_index', 'dataset']
}
MARKET_BENCHMARKS: Dict[str, dict[str, Any]] = {
    name: benchmark
    for name in ['nasdaq100', 'cac40', 'dji', 'eurostoxx50', 'sp500', 'index', 'world_index', 'dataset']
    if (benchmark := get_universe_benchmark(name)) is not None
}
