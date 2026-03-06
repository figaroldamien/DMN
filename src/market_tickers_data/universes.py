"""Ticker universes by market."""

from __future__ import annotations

from typing import Dict, List

from .components import CAC40, DATASET, DATASET_ALL, INDEX, NASDAQ100

# Backward-compat aliases for existing code paths.
TABLE8_ASSETS_BY_CATEGORY: Dict[str, List[str]] = DATASET
TABLE8_ALL: List[str] = DATASET_ALL

TEST: List[str] = ["^FCHI"]
MARKET_TICKERS: Dict[str, List[str]] = {
    "nasdaq100": NASDAQ100,
    "cac40": CAC40,
    "index": INDEX,
    "dataset_all": DATASET_ALL,
    "table8_all": DATASET_ALL,
    "test": TEST,
}
