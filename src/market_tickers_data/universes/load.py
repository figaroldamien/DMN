"""Load self-describing universe JSON files from the project data directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


_DATA_DIR = Path(__file__).resolve().parents[3] / 'data' / 'market_tickers' / 'universes'


def _load_universe_file(filename: str) -> tuple[dict[str, Any], Dict[str, Dict[str, str]]]:
    path = _DATA_DIR / filename
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)

    metadata: dict[str, Any]
    component_rows: list[dict[str, Any]]
    if isinstance(raw, list):
        metadata = {}
        component_rows = raw
    elif isinstance(raw, dict) and isinstance(raw.get('components'), list):
        metadata = {k: v for k, v in raw.items() if k != 'components'}
        component_rows = raw['components']
    else:
        raise ValueError(f'Invalid components file format: {path}')

    out: Dict[str, Dict[str, str]] = {}
    for item in component_rows:
        if not isinstance(item, dict):
            raise ValueError(f'Invalid component entry in {path}: {item!r}')
        ticker = item.get('ticker')
        if not isinstance(ticker, str) or not ticker:
            raise ValueError(f'Missing/invalid ticker in {path}: {item!r}')
        if ticker in out:
            raise ValueError(f"Duplicate ticker '{ticker}' in {path}")
        meta = {k: v for k, v in item.items() if k != 'ticker'}
        out[ticker] = meta
    return metadata, out


_UNIVERSE_FILENAMES = {
    'nasdaq100': 'nasdaq100_components.json',
    'cac40': 'cac40_components.json',
    'dji': 'dji_components.json',
    'sp500': 'sp500_components.json',
    'sbf120': 'sbf120_components.json',
    'eurostoxx50': 'eurostoxx50_components.json',
    'eurostoxx600': 'eurostoxx600_components.json',
    'index': 'index_components.json',
    'dataset': 'dataset_components.json',
    'futures': 'futures_components.json',
    'world_index': 'world_index_components.json',
}

_UNIVERSE_PAYLOADS = {
    name: _load_universe_file(filename)
    for name, filename in _UNIVERSE_FILENAMES.items()
}
UNIVERSE_METADATA: Dict[str, dict[str, Any]] = {
    name: metadata
    for name, (metadata, _components) in _UNIVERSE_PAYLOADS.items()
}


def get_universe_metadata(name: str) -> dict[str, Any]:
    if name not in UNIVERSE_METADATA:
        raise KeyError(f"Unknown universe '{name}'. Allowed values: {sorted(UNIVERSE_METADATA)}")
    return dict(UNIVERSE_METADATA[name])


def get_universe_benchmark(name: str) -> dict[str, Any] | None:
    metadata = get_universe_metadata(name)
    benchmark = metadata.get('benchmark')
    if benchmark is None:
        return None
    if not isinstance(benchmark, dict):
        raise ValueError(f"Invalid benchmark metadata for universe '{name}': {benchmark!r}")
    return dict(benchmark)


NASDAQ100_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['nasdaq100'][1]
CAC40_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['cac40'][1]
DJI_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['dji'][1]
SP500_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['sp500'][1]
SBF120_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['sbf120'][1]
EUROSTOXX50_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['eurostoxx50'][1]
EUROSTOXX600_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['eurostoxx600'][1]
INDEX_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['index'][1]
DATASET_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['dataset'][1]
FUTURES_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['futures'][1]
WORLD_INDEX_COMPONENTS: Dict[str, Dict[str, str]] = _UNIVERSE_PAYLOADS['world_index'][1]

NASDAQ100: List[str] = list(NASDAQ100_COMPONENTS.keys())
CAC40: List[str] = list(CAC40_COMPONENTS.keys())
DJI: List[str] = list(DJI_COMPONENTS.keys())
SP500: List[str] = list(SP500_COMPONENTS.keys())
SBF120: List[str] = list(SBF120_COMPONENTS.keys())
EUROSTOXX50: List[str] = list(EUROSTOXX50_COMPONENTS.keys())
EUROSTOXX600: List[str] = list(EUROSTOXX600_COMPONENTS.keys())
INDEX: List[str] = list(INDEX_COMPONENTS.keys())
FUTURES: List[str] = list(FUTURES_COMPONENTS.keys())
WORLD_INDEX: List[str] = list(WORLD_INDEX_COMPONENTS.keys())

_DATASET_ORDER = ('fx', 'bond', 'index', 'comdty', 'energy')
DATASET: Dict[str, List[str]] = {
    category: [
        ticker
        for ticker, meta in DATASET_COMPONENTS.items()
        if meta.get('category') == category
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
