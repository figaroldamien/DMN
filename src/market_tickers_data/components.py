"""Core market ticker components loaded from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


_DATA_DIR = Path(__file__).with_name("data")


def _load_components(filename: str) -> Dict[str, Dict[str, str]]:
    path = _DATA_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Invalid components file format: {path}")
    out: Dict[str, Dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid component entry in {path}: {item!r}")
        ticker = item.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError(f"Missing/invalid ticker in {path}: {item!r}")
        if ticker in out:
            raise ValueError(f"Duplicate ticker '{ticker}' in {path}")
        meta = {k: v for k, v in item.items() if k != "ticker"}
        out[ticker] = meta
    return out


NASDAQ100_COMPONENTS: Dict[str, Dict[str, str]] = _load_components("nasdaq100_components.json")
CAC40_COMPONENTS: Dict[str, Dict[str, str]] = _load_components("cac40_components.json")
INDEX_COMPONENTS: Dict[str, Dict[str, str]] = _load_components("index_components.json")
DATASET_COMPONENTS: Dict[str, Dict[str, str]] = _load_components("dataset_components.json")

NASDAQ100: List[str] = list(NASDAQ100_COMPONENTS.keys())
CAC40: List[str] = list(CAC40_COMPONENTS.keys())
INDEX: List[str] = list(INDEX_COMPONENTS.keys())

_DATASET_ORDER = ("fx", "bond", "index", "comdty", "energy")
DATASET: Dict[str, List[str]] = {
    category: [
        ticker
        for ticker, meta in DATASET_COMPONENTS.items()
        if meta.get("category") == category
    ]
    for category in _DATASET_ORDER
}

DATASET_ALL: List[str] = list(
    dict.fromkeys(
        ticker
        for category in _DATASET_ORDER
        for ticker in DATASET[category]
    )
)
