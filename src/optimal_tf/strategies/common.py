from __future__ import annotations

import pandas as pd

from ..config import EstimationConfig
from ..estimators.pipeline import estimate_clean_covariance_at_date, estimate_clean_covariance_panel
from ..features import compute_returns, ewma_vol, normalize_returns_by_vol, sanitize_returns
from .types import StrategyState


def resolve_allocation_date(index: pd.Index, as_of_date: str | pd.Timestamp | None = None) -> pd.Timestamp:
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
    returns = sanitize_returns(compute_returns(prices), max_abs_return=est_cfg.max_abs_return)
    vol = ewma_vol(returns, span=est_cfg.vol_span)
    z_returns = normalize_returns_by_vol(returns, vol)
    return returns, vol, z_returns


def weights_to_strategy_state(weights: pd.Series) -> StrategyState:
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
    if covariance_cache:
        eligible = [ts for ts in covariance_cache if ts <= date]
        if eligible:
            return covariance_cache[max(eligible)]
    return estimate_clean_covariance_at_date(prices, est_cfg, date)


def resolve_covariance_cache_until_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    date: pd.Timestamp,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    if covariance_cache:
        subset = {ts: cov for ts, cov in covariance_cache.items() if ts <= date}
        if subset:
            return subset
    history = prices.loc[prices.index <= date]
    return estimate_clean_covariance_panel(history, est_cfg)
