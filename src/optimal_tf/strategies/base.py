from __future__ import annotations

"""Core registry for covariance-based base strategies.

This module only handles the strategies that can be expressed directly as a
function of a cleaned covariance matrix plus simple data-availability rules.
The only non-base strategy family still orchestrated higher up in `api.py` is LLTF.

The distinction matters because:
- EW depends only on which assets have a price on the allocation date
- RP / ARP / NM depend on one cleaned covariance snapshot
- LLTF needs richer intermediate state than a single covariance matrix
"""

from typing import Callable

import pandas as pd

from ..config import EstimationConfig
from .weights import normalize_weights
from .arp import agnostic_risk_parity_weights_from_cov
from .nm import naive_markowitz_weights_from_cov
from .rp import risk_parity_weights_from_cov
from .common import resolve_covariance_at_date


# Only the "pure covariance -> weights" strategies belong in this registry.
BASE_STRATEGY_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "RP": risk_parity_weights_from_cov,
    "ARP": agnostic_risk_parity_weights_from_cov,
    "NM": naive_markowitz_weights_from_cov,
}

ALL_STRATEGIES = sorted([*BASE_STRATEGY_REGISTRY, "EW", "LLTF"])


def strategy_registry() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """Return a shallow copy of the base strategy registry.

    Returning a copy avoids accidental in-place mutation by callers while still
    making the mapping easy to inspect in notebooks and dashboards.
    """
    return dict(BASE_STRATEGY_REGISTRY)


def supported_strategies() -> list[str]:
    """Return the full list of strategy identifiers exposed by the package."""
    return list(ALL_STRATEGIES)


def resolve_strategy(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Look up one base covariance-based strategy by name."""
    if name not in BASE_STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Allowed values: {sorted(BASE_STRATEGY_REGISTRY)}")
    return BASE_STRATEGY_REGISTRY[name]


def compute_base_weights_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.Series:
    """Compute the structural weight vector for one base strategy and one date.

    The output of this function is a *base* allocation, before any higher-level
    strategy-specific logic such as LLTF modifies it.
    """
    # EW is the only base strategy that depends purely on data availability and
    # does not require a covariance estimate.
    if strategy == "EW":
        available = prices.loc[date].dropna().index
        if len(available) == 0:
            return pd.Series(0.0, index=prices.columns, dtype=float)
        row = pd.Series(0.0, index=prices.columns, dtype=float)
        row.loc[available] = 1.0 / len(available)
        return normalize_weights(row, long_only=long_only)

    # RP / ARP / NM all share the same cleaned covariance input and only differ
    # by the portfolio mapping applied to that matrix.
    cov = resolve_covariance_at_date(prices, est_cfg, date, covariance_cache)
    if strategy in BASE_STRATEGY_REGISTRY:
        raw = resolve_strategy(strategy)(cov).reindex(prices.columns).fillna(0.0)
        if long_only:
            raw = normalize_weights(raw, long_only=True)
        return raw.astype(float)

    raise KeyError(f"Unsupported base strategy '{strategy}'.")
