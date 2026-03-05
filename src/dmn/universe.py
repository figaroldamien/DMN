from __future__ import annotations

import warnings
from typing import Dict, List

from market_tickers import (
    CAC40_COMPONENTS,
    MARKET_TICKERS,
    NASDAQ100_COMPONENTS,
    tickers_by_sector,
    tickers_by_sector_and_subsector,
)


def resolve_tickers(market: str, sector: str | None = None, sub_sector: str | None = None) -> List[str]:
    tickers = MARKET_TICKERS[market]
    component_indices: Dict[str, Dict[str, Dict[str, str]]] = {
        "nasdaq100": NASDAQ100_COMPONENTS,
        "cac40": CAC40_COMPONENTS,
    }

    if market in component_indices:
        components = component_indices[market]
        if sector and sub_sector:
            tickers = tickers_by_sector_and_subsector(components, sector, sub_sector)
        elif sector:
            tickers = tickers_by_sector(components, sector)
        elif sub_sector:
            sub_sector_norm = sub_sector.strip().lower()
            tickers = [
                ticker
                for ticker, meta in components.items()
                if meta.get("sub_sector", "").strip().lower() == sub_sector_norm
            ]
    elif sector or sub_sector:
        warnings.warn(
            f"--sector/--sub-sector ignored for market '{market}' "
            "because this universe has no component metadata."
        )

    if not tickers:
        raise ValueError("No tickers selected after applying market/sector/sub-sector filters.")

    return tickers
