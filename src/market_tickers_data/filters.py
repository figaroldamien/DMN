"""Ticker filtering helpers."""

from __future__ import annotations

from typing import Dict, List


def component_filters_catalog(
    components: Dict[str, Dict[str, str]],
) -> Dict[str, List[str]]:
    """
    Return distinct filter values available in a component dictionary.

    The returned dictionary always contains:
    - "categories"
    - "sectors"
    - "sub_sectors"
    """
    categories = sorted(
        {
            meta.get("category", "").strip()
            for meta in components.values()
            if meta.get("category", "").strip()
        }
    )
    sectors = sorted(
        {
            meta.get("sector", "").strip()
            for meta in components.values()
            if meta.get("sector", "").strip()
        }
    )
    sub_sectors = sorted(
        {
            meta.get("sub_sector", "").strip()
            for meta in components.values()
            if meta.get("sub_sector", "").strip()
        }
    )
    return {
        "categories": categories,
        "sectors": sectors,
        "sub_sectors": sub_sectors,
    }


def tickers_by_sector_and_subsector(
    index_components: Dict[str, Dict[str, str]],
    sector: str,
    sub_sector: str,
) -> List[str]:
    """Return tickers matching both sector and sub-sector."""
    sector_norm = sector.strip().lower()
    sub_sector_norm = sub_sector.strip().lower()
    return [
        ticker
        for ticker, meta in index_components.items()
        if meta.get("sector", "").strip().lower() == sector_norm
        and meta.get("sub_sector", "").strip().lower() == sub_sector_norm
    ]


def tickers_by_sector(
    index_components: Dict[str, Dict[str, str]],
    sector: str,
) -> List[str]:
    """Return tickers matching a sector."""
    sector_norm = sector.strip().lower()
    return [
        ticker
        for ticker, meta in index_components.items()
        if meta.get("sector", "").strip().lower() == sector_norm
    ]


def tickers_by_category(
    components: Dict[str, Dict[str, str]],
    category: str,
) -> List[str]:
    """Return tickers matching a category."""
    category_norm = category.strip().lower()
    return [
        ticker
        for ticker, meta in components.items()
        if meta.get("category", "").strip().lower() == category_norm
    ]
