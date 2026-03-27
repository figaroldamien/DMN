from __future__ import annotations

from typing import Callable

import pandas as pd

from .backtest import build_weight_panel
from .config import EstimationConfig
from .estimators.pipeline import estimate_clean_covariance_panel
from .features import compute_returns, ewma_vol, normalize_returns_by_vol, sanitize_returns, trend_ema_signal
from .portfolios import (
    agnostic_risk_parity_weights_from_cov,
    naive_markowitz_weights_from_cov,
    normalize_weights,
    risk_parity_weights_from_cov,
)


def strategy_registry() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "RP": risk_parity_weights_from_cov,
        "ARP": agnostic_risk_parity_weights_from_cov,
        "NM": naive_markowitz_weights_from_cov,
    }


def supported_strategies() -> list[str]:
    return sorted([*strategy_registry(), "EW", "ToRP"])


def resolve_strategy(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    registry = strategy_registry()
    if name not in registry:
        raise KeyError(f"Unknown strategy '{name}'. Allowed values: {sorted(registry)}")
    return registry[name]


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


def _equal_weight_panel(prices: pd.DataFrame, *, long_only: bool) -> pd.DataFrame:
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ts in prices.index:
        available = prices.loc[ts].dropna().index
        if len(available) == 0:
            continue
        row = pd.Series(0.0, index=prices.columns, dtype=float)
        row.loc[available] = 1.0 / len(available)
        weights.loc[ts] = normalize_weights(row, long_only=long_only)
    return weights.ffill().fillna(0.0)


def _trend_on_risk_parity_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    long_only: bool,
) -> pd.DataFrame:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=est_cfg.max_abs_return)
    vol = ewma_vol(returns, span=est_cfg.vol_span)
    z_returns = normalize_returns_by_vol(returns, vol)
    trend = trend_ema_signal(z_returns, span=est_cfg.trend_span)
    cov_panel = estimate_clean_covariance_panel(prices, est_cfg)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for ts, cov in cov_panel.items():
        base = risk_parity_weights_from_cov(cov)
        signal = trend.loc[ts].reindex(base.index).fillna(0.0)
        # First V1 ToRP: use RP as the risk budget and let the trend estimate
        # decide the sign and relative conviction across assets.
        raw = base * signal
        row = pd.Series(0.0, index=prices.columns, dtype=float)
        row.loc[raw.index] = normalize_weights(raw, long_only=long_only)
        weights.loc[ts] = row
    return weights.ffill().fillna(0.0)


def compute_weights_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    long_only: bool = False,
) -> pd.DataFrame:
    if strategy == "EW":
        return _equal_weight_panel(prices, long_only=long_only)
    if strategy == "ToRP":
        return _trend_on_risk_parity_panel(prices, est_cfg, long_only=long_only)
    weight_fn = resolve_strategy(strategy)
    return build_weight_panel(prices, est_cfg, weight_fn, long_only=long_only)


def compute_portfolio_weights_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    long_only: bool = False,
) -> tuple[pd.Timestamp, pd.Series]:
    weights = compute_weights_panel(prices, est_cfg, strategy, long_only=long_only)
    date = resolve_allocation_date(weights.index, as_of_date=as_of_date)
    return date, weights.loc[date].dropna()
