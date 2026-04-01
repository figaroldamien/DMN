from __future__ import annotations

from typing import Callable

import pandas as pd

from ..config import EstimationConfig
from ..portfolios import (
    agnostic_risk_parity_weights_from_cov,
    naive_markowitz_weights_from_cov,
    normalize_weights,
    risk_parity_weights_from_cov,
)
from .common import resolve_covariance_at_date


BASE_STRATEGY_REGISTRY: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "RP": risk_parity_weights_from_cov,
    "ARP": agnostic_risk_parity_weights_from_cov,
    "NM": naive_markowitz_weights_from_cov,
}

ALL_STRATEGIES = sorted([*BASE_STRATEGY_REGISTRY, "EW", "LLTF", "ToRP0", "ToRP1", "ToRP2", "ToRP3"])


def strategy_registry() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return dict(BASE_STRATEGY_REGISTRY)


def supported_strategies() -> list[str]:
    return list(ALL_STRATEGIES)


def resolve_strategy(name: str) -> Callable[[pd.DataFrame], pd.Series]:
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
