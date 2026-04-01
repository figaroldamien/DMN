from __future__ import annotations

import pandas as pd

from ..config import EstimationConfig
from .base import compute_base_weights_at_date, strategy_registry, supported_strategies
from .common import resolve_allocation_date, weights_to_strategy_state
from .lltf import lead_lag_trend_following_at_date
from .torp import (
    build_torp_factor_context,
    trend_on_risk_parity_v0_at_date,
    trend_on_risk_parity_v1_at_date,
    trend_on_risk_parity_v2_at_date,
    trend_on_risk_parity_v3_at_date,
)
from .types import StrategyPanel, StrategyState


_TORP_STATE_BUILDERS = {
    "ToRP0": trend_on_risk_parity_v0_at_date,
    "ToRP1": trend_on_risk_parity_v1_at_date,
    "ToRP2": trend_on_risk_parity_v2_at_date,
    "ToRP3": trend_on_risk_parity_v3_at_date,
}


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
    ts = resolve_allocation_date(prices.index, as_of_date=date)
    if strategy in _TORP_STATE_BUILDERS:
        kwargs = {
            "date": ts,
            "long_only": long_only,
            "covariance_cache": covariance_cache,
        }
        if strategy in {"ToRP2", "ToRP3"}:
            kwargs["torp_context"] = strategy_context
        return _TORP_STATE_BUILDERS[strategy](prices, est_cfg, **kwargs)
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
    target_index = pd.DatetimeIndex(prices.index if target_dates is None else target_dates)
    base_weights = pd.DataFrame(0.0, index=target_index, columns=prices.columns, dtype=float)
    effective_weights = pd.DataFrame(0.0, index=target_index, columns=prices.columns, dtype=float)
    signal_scale = pd.Series(0.0, index=target_index, dtype=float)
    strategy_context = None
    if strategy in {"ToRP2", "ToRP3"} and covariance_cache:
        strategy_context = build_torp_factor_context(prices, est_cfg, covariance_cache)
    for ts in target_index:
        try:
            state = compute_strategy_state_at_date(
                prices,
                est_cfg,
                strategy,
                date=ts,
                long_only=long_only,
                covariance_cache=covariance_cache,
                strategy_context=strategy_context,
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
    date = resolve_allocation_date(prices.index, as_of_date=as_of_date)
    state = compute_strategy_state_at_date(prices, est_cfg, strategy, date=date, long_only=long_only)
    return date, state
