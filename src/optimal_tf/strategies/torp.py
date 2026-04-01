from __future__ import annotations

import pandas as pd

from market_tickers_data.components import (
    CAC40_COMPONENTS,
    DATASET_COMPONENTS,
    INDEX_COMPONENTS,
    NASDAQ100_COMPONENTS,
)

from ..config import EstimationConfig
from ..features import ewma_vol, trend_ema_signal
from ..portfolios import (
    risk_parity_weights_from_cov,
    risk_parity_weights_from_cov_with_tilt,
    trend_on_risk_parity_weights_from_cov_and_factor_signal,
    trend_on_risk_parity_weights_from_cov_and_signal,
)
from .common import (
    resolve_covariance_at_date,
    resolve_covariance_cache_until_date,
    sanitized_normalized_returns,
)
from .types import StrategyState


_ALL_COMPONENTS = {}
_ALL_COMPONENTS.update(DATASET_COMPONENTS)
_ALL_COMPONENTS.update(INDEX_COMPONENTS)
_ALL_COMPONENTS.update(CAC40_COMPONENTS)
_ALL_COMPONENTS.update(NASDAQ100_COMPONENTS)


def torp_rp_tilt(tickers: pd.Index) -> pd.Series:
    # ToRP2/3 follow the paper's convention of excluding FX from the RP factor
    # when that metadata is available in the ticker registry.
    tilt = pd.Series(1.0, index=tickers, dtype=float)
    for ticker in tickers:
        meta = _ALL_COMPONENTS.get(str(ticker), {})
        if meta.get("category") == "fx":
            tilt.loc[ticker] = 0.0
    return tilt


def build_torp_factor_context(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame],
) -> dict[str, pd.DataFrame | pd.Series]:
    # The expensive part of ToRP2/3 is reconstructing the historical RP factor.
    # We do it once per run, then reuse:
    # - the tilted RP base weights
    # - the raw factor trend signal (ToRP2)
    # - the vol-normalized factor signal (ToRP3)
    _, _, z_returns = sanitized_normalized_returns(prices, est_cfg)
    tilt = torp_rp_tilt(prices.columns)
    base_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns, dtype=float)
    for ts, cov in covariance_cache.items():
        if ts not in base_weights.index:
            continue
        base = risk_parity_weights_from_cov_with_tilt(cov, tilt)
        base_weights.loc[ts, base.index] = base
    base_weights = base_weights.ffill().fillna(0.0)
    rp_factor_returns = (base_weights * z_returns.fillna(0.0)).sum(axis=1)
    rp_factor_signal_v2 = trend_ema_signal(
        rp_factor_returns.to_frame("rp"),
        alpha=est_cfg.trend_alpha,
        span=est_cfg.trend_span,
    )["rp"]
    rp_factor_vol = ewma_vol(
        rp_factor_returns.to_frame("rp"),
        alpha=est_cfg.trend_alpha,
        span=est_cfg.trend_span,
    )["rp"]
    rp_factor_signal_v3 = (rp_factor_signal_v2 / (rp_factor_vol + 1e-12)).replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )
    return {
        "base_weights": base_weights,
        "signal_v2": rp_factor_signal_v2,
        "signal_v3": rp_factor_signal_v3,
    }


def trend_on_risk_parity_v0_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyState:
    # ToRP0 is the legacy implementation: compute RP weights, apply asset-level
    # trend signs, then renormalize the resulting cross-section.
    history = prices.loc[prices.index <= date]
    _, _, z_returns = sanitized_normalized_returns(history, est_cfg)
    trend = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    cov = resolve_covariance_at_date(history, est_cfg, date, covariance_cache)
    base = risk_parity_weights_from_cov(cov).reindex(prices.columns).fillna(0.0)
    signal = trend.loc[date].reindex(base.index).fillna(0.0)
    from ..portfolios import normalize_weights

    effective = normalize_weights(base * signal, long_only=long_only).reindex(prices.columns).fillna(0.0)
    return StrategyState(base_weights=base, signal_scale=1.0, effective_weights=effective)


def trend_on_risk_parity_v1_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyState:
    # ToRP1 keeps the RP base portfolio but reduces the signal to a single
    # projected scalar. The strategy remains fully renormalized afterwards.
    history = prices.loc[prices.index <= date]
    _, _, z_returns = sanitized_normalized_returns(history, est_cfg)
    trend = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    cov = resolve_covariance_at_date(history, est_cfg, date, covariance_cache)
    base = risk_parity_weights_from_cov(cov).reindex(prices.columns).fillna(0.0)
    effective = trend_on_risk_parity_weights_from_cov_and_signal(
        cov,
        trend.loc[date],
        long_only=long_only,
    ).reindex(prices.columns).fillna(0.0)
    projected_signal = 0.0 if base.abs().sum() == 0 else float(effective @ base)
    return StrategyState(base_weights=base, signal_scale=projected_signal, effective_weights=effective)


def trend_on_risk_parity_v2_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    torp_context: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> StrategyState:
    # ToRP2 aligns with the paper more closely by computing trend on the RP
    # factor itself, then applying that common factor signal to the tilted RP
    # portfolio.
    tilt = torp_rp_tilt(prices.columns)
    history = prices.loc[prices.index <= date]
    if torp_context is None:
        cov_panel = resolve_covariance_cache_until_date(history, est_cfg, date, covariance_cache)
        torp_context = build_torp_factor_context(history, est_cfg, cov_panel)
    cov = resolve_covariance_at_date(history, est_cfg, date, covariance_cache)
    base = risk_parity_weights_from_cov_with_tilt(cov, tilt).reindex(prices.columns).fillna(0.0)
    rp_factor_signal = torp_context["signal_v2"]
    factor_signal = 0.0 if pd.isna(rp_factor_signal.loc[date]) else float(rp_factor_signal.loc[date])
    effective = trend_on_risk_parity_weights_from_cov_and_factor_signal(
        cov,
        factor_signal,
        tilt=tilt,
        long_only=long_only,
    ).reindex(prices.columns).fillna(0.0)
    return StrategyState(base_weights=base, signal_scale=factor_signal, effective_weights=effective)


def trend_on_risk_parity_v3_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    torp_context: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> StrategyState:
    # ToRP3 is the first variant that preserves signal amplitude explicitly:
    # effective exposure is "base RP portfolio x factor signal" rather than a
    # renormalized portfolio with the same gross exposure every period.
    history = prices.loc[prices.index <= date]
    if torp_context is None:
        cov_panel = resolve_covariance_cache_until_date(history, est_cfg, date, covariance_cache)
        torp_context = build_torp_factor_context(history, est_cfg, cov_panel)
    base_weights = torp_context["base_weights"]
    raw_signal = torp_context["signal_v3"].loc[date]
    factor_signal = 0.0 if pd.isna(raw_signal) else float(raw_signal)
    factor_signal *= float(est_cfg.torp_signal_gain)
    base = base_weights.loc[date].reindex(prices.columns).fillna(0.0)
    effective = base * factor_signal
    if long_only:
        effective = effective.clip(lower=0.0)
    return StrategyState(base_weights=base, signal_scale=factor_signal, effective_weights=effective)
