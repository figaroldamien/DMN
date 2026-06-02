from __future__ import annotations

"""Public entrypoint for the `optimal_tf.strategies` package.

This package mixes three layers that are useful to expose from one place:
- low-level weight builders such as RP / ARP / NM
- orchestration helpers that turn one-date builders into dated strategy states
- small shared utilities such as date resolution and normalization helpers

Keeping these re-exports centralized makes notebooks, dashboards, and services
less verbose while still letting the implementation live in focused modules.
"""

from .weights import normalize_weights
from .arp import agnostic_risk_parity_weights_from_cov
from .nm import naive_markowitz_weights_from_cov
from .rp import risk_parity_weights_from_cov
from .api import (
    compute_portfolio_strategy_state_at_date,
    compute_portfolio_weights_at_date,
    compute_strategy_panel,
    compute_strategy_state_at_date,
    compute_weights_panel,
)
from .base import resolve_strategy, strategy_registry, supported_strategies
from .common import resolve_allocation_date
from .lltf import _lead_lag_virtual_returns
from .types import StrategyPanel, StrategyState

__all__ = [
    "StrategyPanel",
    "StrategyState",
    "_lead_lag_virtual_returns",
    "agnostic_risk_parity_weights_from_cov",
    "naive_markowitz_weights_from_cov",
    "risk_parity_weights_from_cov",
    "normalize_weights",
    "compute_portfolio_strategy_state_at_date",
    "compute_portfolio_weights_at_date",
    "compute_strategy_panel",
    "compute_strategy_state_at_date",
    "compute_weights_panel",
    "resolve_allocation_date",
    "resolve_strategy",
    "strategy_registry",
    "supported_strategies",
]
