from __future__ import annotations

"""Shared helpers used by several strategy families.

These utilities are intentionally kept strategy-agnostic. They mostly solve
small recurring problems:
- choose a valid allocation date from a market index
- build a clean `returns / vol / z-returns` bundle from prices
- wrap a simple weight vector into the common `StrategyState` shape
- reuse covariance snapshots when a caller already built a cache
"""

import pandas as pd

from ..config import EstimationConfig
from trading_core.risk import (
    covariance_to_correlation,
    estimate_clean_correlation_at_date,
    estimate_clean_correlation_panel,
    estimate_clean_covariance_at_date,
    estimate_clean_covariance_panel,
)
from ..features import compute_returns, ewma_vol, normalize_returns_by_vol, sanitize_returns
from .types import StrategyState

_MAX_CACHE_STALENESS_DAYS = 7


def resolve_allocation_date(index: pd.Index, as_of_date: str | pd.Timestamp | None = None) -> pd.Timestamp:
    """Resolve the last available market date on or before the requested date.

    The dashboard and CLI often ask for an allocation "as of" a calendar date.
    This helper converts that request into the most recent actual observation in
    the provided market index.
    """
    if len(index) == 0:
        raise ValueError("Cannot resolve an allocation date on an empty index.")
    if as_of_date is None:
        target = pd.Timestamp.today().normalize()
    else:
        target = pd.Timestamp(as_of_date).normalize()
    eligible = pd.Index(index[index <= target])
    if len(eligible) == 0:
        raise ValueError(f"No data available on or before {target.date()}.")
    return pd.Timestamp(eligible.max())


def sanitized_normalized_returns(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the return bundle reused by trend-based strategies.

    Returns three aligned objects:
    - raw arithmetic returns after outlier sanitization
    - EWMA volatility estimates
    - volatility-normalized returns (`z_returns`)

    LLTF and any future signal-based strategy can rely on this exact pipeline, so centralizing it prevents
    subtle drift between implementations.
    """
    returns = sanitize_returns(compute_returns(prices), max_abs_return=est_cfg.max_abs_return)
    vol = ewma_vol(returns, span=est_cfg.vol_span)
    z_returns = normalize_returns_by_vol(returns, vol)
    return returns, vol, z_returns


def weights_to_strategy_state(weights: pd.Series) -> StrategyState:
    """Wrap a plain weight vector into the package-wide state structure.

    This is used by simple strategies where the effective portfolio is exactly
    the base portfolio and there is no separate timing signal amplitude.
    """
    weights = weights.astype(float)
    return StrategyState(
        base_weights=weights.copy(),
        signal_scale=1.0,
        effective_weights=weights.copy(),
    )


def resolve_covariance_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    date: pd.Timestamp,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return the cleaned covariance snapshot to use on one allocation date.

    If a cache is provided, we only reuse snapshots that are still fresh around
    the requested date. This prevents month- or year-long freezes when a
    cleaner fails to produce matrices for a stretch of rebalance dates.
    """
    if covariance_cache:
        if date in covariance_cache:
            return covariance_cache[date]
        eligible = [ts for ts in covariance_cache if ts <= date]
        if eligible:
            latest = max(eligible)
            if (date - latest).days <= _MAX_CACHE_STALENESS_DAYS:
                return covariance_cache[latest]
    return estimate_clean_covariance_at_date(prices, est_cfg, date)


def resolve_clean_correlation_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    date: pd.Timestamp,
    correlation_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return the cleaned correlation snapshot to use on one allocation date."""
    if correlation_cache:
        if date in correlation_cache:
            return correlation_cache[date]
        eligible = [ts for ts in correlation_cache if ts <= date]
        if eligible:
            latest = max(eligible)
            if (date - latest).days <= _MAX_CACHE_STALENESS_DAYS:
                return correlation_cache[latest]
    if covariance_cache:
        if date in covariance_cache:
            cached = covariance_cache[date]
            diag = cached.to_numpy(dtype=float).diagonal()
            if len(diag) > 0 and (abs(diag - 1.0) < 1e-6).all():
                return cached.astype(float)
            return covariance_to_correlation(cached)
        eligible = [ts for ts in covariance_cache if ts <= date]
        if eligible:
            latest = max(eligible)
            if (date - latest).days <= _MAX_CACHE_STALENESS_DAYS:
                cached = covariance_cache[latest]
                diag = cached.to_numpy(dtype=float).diagonal()
                if len(diag) > 0 and (abs(diag - 1.0) < 1e-6).all():
                    return cached.astype(float)
                return covariance_to_correlation(cached)
    return estimate_clean_correlation_at_date(prices, est_cfg, date)


def resolve_covariance_cache_until_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    date: pd.Timestamp,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Return a covariance cache truncated to the requested history.

    This helper is useful for any strategy that needs a whole panel of
    historical covariance estimates rather than just one dated snapshot.
    """
    if covariance_cache:
        subset = {ts: cov for ts, cov in covariance_cache.items() if ts <= date}
        if subset:
            return subset
    history = prices.loc[prices.index <= date]
    return estimate_clean_covariance_panel(history, est_cfg)


def resolve_clean_correlation_cache_until_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    date: pd.Timestamp,
    correlation_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Return a cleaned-correlation cache truncated to the requested history."""
    if correlation_cache:
        subset = {ts: corr for ts, corr in correlation_cache.items() if ts <= date}
        if subset:
            return subset
    if covariance_cache:
        subset = {ts: covariance_to_correlation(cov) for ts, cov in covariance_cache.items() if ts <= date}
        if subset:
            return subset
    history = prices.loc[prices.index <= date]
    return estimate_clean_correlation_panel(history, est_cfg)
