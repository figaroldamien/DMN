from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from market_tickers_data.components import (
    CAC40_COMPONENTS,
    DATASET_COMPONENTS,
    INDEX_COMPONENTS,
    NASDAQ100_COMPONENTS,
)

from .backtest import build_weight_panel
from .config import EstimationConfig
from .estimators.pipeline import estimate_clean_covariance_panel
from .features import compute_returns, ewma_vol, normalize_returns_by_vol, sanitize_returns, trend_ema_signal
from .portfolios import (
    agnostic_risk_parity_weights_from_cov,
    naive_markowitz_weights_from_cov,
    normalize_weights,
    risk_parity_weights_from_cov,
    risk_parity_weights_from_cov_with_tilt,
    trend_on_risk_parity_weights_from_cov_and_factor_signal,
    trend_on_risk_parity_weights_from_cov_and_signal,
)


_ALL_COMPONENTS = {}
_ALL_COMPONENTS.update(DATASET_COMPONENTS)
_ALL_COMPONENTS.update(INDEX_COMPONENTS)
_ALL_COMPONENTS.update(CAC40_COMPONENTS)
_ALL_COMPONENTS.update(NASDAQ100_COMPONENTS)


@dataclass(frozen=True)
class StrategyState:
    base_weights: pd.Series
    signal_scale: float
    effective_weights: pd.Series


@dataclass(frozen=True)
class StrategyPanel:
    base_weights: pd.DataFrame
    signal_scale: pd.Series
    effective_weights: pd.DataFrame


def strategy_registry() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "RP": risk_parity_weights_from_cov,
        "ARP": agnostic_risk_parity_weights_from_cov,
        "NM": naive_markowitz_weights_from_cov,
    }


def supported_strategies() -> list[str]:
    return sorted([*strategy_registry(), "EW", "ToRP0", "ToRP1", "ToRP2", "ToRP3"])


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


def _wrap_weights_panel_as_strategy_panel(weights: pd.DataFrame) -> StrategyPanel:
    weights = weights.astype(float)
    return StrategyPanel(
        base_weights=weights.copy(),
        signal_scale=pd.Series(1.0, index=weights.index, dtype=float),
        effective_weights=weights.copy(),
    )


def _sanitized_normalized_returns(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=est_cfg.max_abs_return)
    vol = ewma_vol(returns, span=est_cfg.vol_span)
    z_returns = normalize_returns_by_vol(returns, vol)
    return returns, vol, z_returns


def _trend_on_risk_parity_v0_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    long_only: bool,
) -> pd.DataFrame:
    _, _, z_returns = _sanitized_normalized_returns(prices, est_cfg)
    trend = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    cov_panel = estimate_clean_covariance_panel(prices, est_cfg)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for ts, cov in cov_panel.items():
        base = risk_parity_weights_from_cov(cov)
        signal = trend.loc[ts].reindex(base.index).fillna(0.0)
        raw = base * signal
        row = pd.Series(0.0, index=prices.columns, dtype=float)
        row.loc[raw.index] = normalize_weights(raw, long_only=long_only)
        weights.loc[ts] = row
    return weights.ffill().fillna(0.0)


def _trend_on_risk_parity_v1_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    long_only: bool,
) -> pd.DataFrame:
    _, _, z_returns = _sanitized_normalized_returns(prices, est_cfg)
    trend = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    cov_panel = estimate_clean_covariance_panel(prices, est_cfg)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for ts, cov in cov_panel.items():
        row = pd.Series(0.0, index=prices.columns, dtype=float)
        signal = trend.loc[ts]
        raw = trend_on_risk_parity_weights_from_cov_and_signal(
            cov,
            signal,
            long_only=long_only,
        )
        row.loc[raw.index] = raw
        weights.loc[ts] = row
    return weights.ffill().fillna(0.0)


def _torp_rp_tilt(tickers: pd.Index) -> pd.Series:
    tilt = pd.Series(1.0, index=tickers, dtype=float)
    for ticker in tickers:
        meta = _ALL_COMPONENTS.get(str(ticker), {})
        if meta.get("category") == "fx":
            tilt.loc[ticker] = 0.0
    return tilt


def _trend_on_risk_parity_v2_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    long_only: bool,
) -> pd.DataFrame:
    _, _, z_returns = _sanitized_normalized_returns(prices, est_cfg)
    cov_panel = estimate_clean_covariance_panel(prices, est_cfg)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    rp_base = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    tilt = _torp_rp_tilt(prices.columns)

    for ts, cov in cov_panel.items():
        base = risk_parity_weights_from_cov_with_tilt(cov, tilt)
        rp_base.loc[ts, base.index] = base

    rp_factor_returns = (rp_base.ffill().fillna(0.0) * z_returns.fillna(0.0)).sum(axis=1)
    rp_factor_signal = trend_ema_signal(
        rp_factor_returns.to_frame("rp"),
        alpha=est_cfg.trend_alpha,
        span=est_cfg.trend_span,
    )["rp"]

    for ts, cov in cov_panel.items():
        row = pd.Series(0.0, index=prices.columns, dtype=float)
        factor_signal = 0.0 if pd.isna(rp_factor_signal.loc[ts]) else float(rp_factor_signal.loc[ts])
        raw = trend_on_risk_parity_weights_from_cov_and_factor_signal(
            cov,
            factor_signal,
            tilt=tilt,
            long_only=long_only,
        )
        row.loc[raw.index] = raw
        weights.loc[ts] = row
    return weights.ffill().fillna(0.0)


def _trend_on_risk_parity_v3_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    long_only: bool,
) -> StrategyPanel:
    _, _, z_returns = _sanitized_normalized_returns(prices, est_cfg)
    cov_panel = estimate_clean_covariance_panel(prices, est_cfg)
    base_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    effective_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    signal_scale = pd.Series(0.0, index=prices.index, dtype=float)
    tilt = _torp_rp_tilt(prices.columns)

    for ts, cov in cov_panel.items():
        base = risk_parity_weights_from_cov_with_tilt(cov, tilt)
        base_weights.loc[ts, base.index] = base

    rp_factor_returns = (base_weights.ffill().fillna(0.0) * z_returns.fillna(0.0)).sum(axis=1)
    rp_factor_signal_raw = trend_ema_signal(
        rp_factor_returns.to_frame("rp"),
        alpha=est_cfg.trend_alpha,
        span=est_cfg.trend_span,
    )["rp"]
    rp_factor_vol = ewma_vol(
        rp_factor_returns.to_frame("rp"),
        alpha=est_cfg.trend_alpha,
        span=est_cfg.trend_span,
    )["rp"]
    rp_factor_signal = (rp_factor_signal_raw / (rp_factor_vol + 1e-12)).replace([float("inf"), float("-inf")], pd.NA)

    for ts in prices.index:
        base = base_weights.loc[ts].reindex(prices.columns).fillna(0.0)
        factor_signal = 0.0 if pd.isna(rp_factor_signal.loc[ts]) else float(rp_factor_signal.loc[ts])
        factor_signal *= float(est_cfg.torp_signal_gain)
        effective = base * factor_signal
        if long_only:
            effective = effective.clip(lower=0.0)
        effective_weights.loc[ts] = effective
        signal_scale.loc[ts] = factor_signal

    return StrategyPanel(
        base_weights=base_weights.ffill().fillna(0.0),
        signal_scale=signal_scale.ffill().fillna(0.0),
        effective_weights=effective_weights.ffill().fillna(0.0),
    )


def compute_strategy_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    long_only: bool = False,
) -> StrategyPanel:
    if strategy == "ToRP3":
        return _trend_on_risk_parity_v3_panel(prices, est_cfg, long_only=long_only)
    weights = compute_weights_panel(prices, est_cfg, strategy, long_only=long_only)
    return _wrap_weights_panel_as_strategy_panel(weights)


def compute_weights_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    long_only: bool = False,
) -> pd.DataFrame:
    if strategy == "EW":
        return _equal_weight_panel(prices, long_only=long_only)
    if strategy == "ToRP0":
        return _trend_on_risk_parity_v0_panel(prices, est_cfg, long_only=long_only)
    if strategy == "ToRP1":
        return _trend_on_risk_parity_v1_panel(prices, est_cfg, long_only=long_only)
    if strategy == "ToRP2":
        return _trend_on_risk_parity_v2_panel(prices, est_cfg, long_only=long_only)
    if strategy == "ToRP3":
        return _trend_on_risk_parity_v3_panel(prices, est_cfg, long_only=long_only).effective_weights
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


def compute_portfolio_strategy_state_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    long_only: bool = False,
) -> tuple[pd.Timestamp, StrategyState]:
    panel = compute_strategy_panel(prices, est_cfg, strategy, long_only=long_only)
    date = resolve_allocation_date(panel.effective_weights.index, as_of_date=as_of_date)
    return date, StrategyState(
        base_weights=panel.base_weights.loc[date].dropna(),
        signal_scale=float(panel.signal_scale.loc[date]),
        effective_weights=panel.effective_weights.loc[date].dropna(),
    )
