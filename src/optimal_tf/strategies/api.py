from __future__ import annotations

"""High-level strategy orchestration helpers.

The modules `rp.py`, `arp.py`, `nm.py`, and `lltf.py` each know how to build
one strategy state at one date. This file provides the bridge between those
single-date engines and the rest of the application, which often needs:
- a single dated portfolio snapshot
- a full panel of rebalance states across time
- a simplified `weights only` view for backtests and reports

The strategy layer is now intentionally simple:
- `LLTF` has a dedicated dated builder because it estimates a dynamic signal
- all other supported strategies are treated as direct base allocations
"""

import pandas as pd

from ..config import EstimationConfig
from .base import compute_base_weights_at_date
from .common import resolve_allocation_date, weights_to_strategy_state
from .lltf import lead_lag_trend_following_at_date
from .types import StrategyPanel, StrategyState


def compute_strategy_state_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    date: pd.Timestamp | str,
    long_only: bool = False,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    strategy_context: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> StrategyState:
    """Compute the full dated state of one strategy."""
    del strategy_context
    ts = resolve_allocation_date(prices.index, as_of_date=date)

    if strategy == "LLTF":
        return lead_lag_trend_following_at_date(prices, est_cfg, date=ts, long_only=long_only)

    return weights_to_strategy_state(
        compute_base_weights_at_date(
            prices,
            est_cfg,
            strategy,
            date=ts,
            long_only=long_only,
            covariance_cache=covariance_cache,
        )
    )


def compute_strategy_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    long_only: bool = False,
    target_dates: pd.Index | None = None,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyPanel:
    """Compute a full panel of dated strategy states."""
    target_index = pd.DatetimeIndex(prices.index if target_dates is None else target_dates)
    base_weights = pd.DataFrame(0.0, index=target_index, columns=prices.columns, dtype=float)
    effective_weights = pd.DataFrame(0.0, index=target_index, columns=prices.columns, dtype=float)
    signal_scale = pd.Series(0.0, index=target_index, dtype=float)

    for ts in target_index:
        try:
            state = compute_strategy_state_at_date(
                prices,
                est_cfg,
                strategy,
                date=ts,
                long_only=long_only,
                covariance_cache=covariance_cache,
            )
        except ValueError:
            continue
        base_weights.loc[ts] = state.base_weights.reindex(prices.columns).fillna(0.0)
        effective_weights.loc[ts] = state.effective_weights.reindex(prices.columns).fillna(0.0)
        signal_scale.loc[ts] = float(state.signal_scale)

    return StrategyPanel(base_weights=base_weights, signal_scale=signal_scale, effective_weights=effective_weights)


def compute_weights_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    long_only: bool = False,
    target_dates: pd.Index | None = None,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return only the effective-weight panel for a strategy."""
    return compute_strategy_panel(
        prices,
        est_cfg,
        strategy,
        long_only=long_only,
        target_dates=target_dates,
        covariance_cache=covariance_cache,
    ).effective_weights


def compute_portfolio_weights_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    long_only: bool = False,
) -> tuple[pd.Timestamp, pd.Series]:
    """Resolve one allocation date and return the final traded weights."""
    date = resolve_allocation_date(prices.index, as_of_date=as_of_date)
    state = compute_strategy_state_at_date(prices, est_cfg, strategy, date=date, long_only=long_only)
    return date, state.effective_weights.dropna()


def compute_portfolio_strategy_state_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    long_only: bool = False,
) -> tuple[pd.Timestamp, StrategyState]:
    """Resolve one allocation date and return the full `StrategyState`."""
    date = resolve_allocation_date(prices.index, as_of_date=as_of_date)
    state = compute_strategy_state_at_date(prices, est_cfg, strategy, date=date, long_only=long_only)
    return date, state
