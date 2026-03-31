from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from market_tickers_data.components import (
    CAC40_COMPONENTS,
    DATASET_COMPONENTS,
    INDEX_COMPONENTS,
    NASDAQ100_COMPONENTS,
)

from .backtest import build_weight_panel
from .config import EstimationConfig
from .estimators.pipeline import estimate_clean_covariance_at_date, estimate_clean_covariance_panel
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
    return sorted([*strategy_registry(), "EW", "LLTF", "ToRP0", "ToRP1", "ToRP2", "ToRP3"])


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


def _weights_to_strategy_state(weights: pd.Series) -> StrategyState:
    weights = weights.astype(float)
    return StrategyState(
        base_weights=weights.copy(),
        signal_scale=1.0,
        effective_weights=weights.copy(),
    )


def _resolve_covariance_at_date(
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


def _resolve_covariance_cache_until_date(
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


def _build_torp_factor_context(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame],
) -> dict[str, pd.DataFrame | pd.Series]:
    _, _, z_returns = _sanitized_normalized_returns(prices, est_cfg)
    tilt = _torp_rp_tilt(prices.columns)
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
    rp_factor_signal_v3 = (rp_factor_signal_v2 / (rp_factor_vol + 1e-12)).replace([float("inf"), float("-inf")], pd.NA)
    return {
        "base_weights": base_weights,
        "signal_v2": rp_factor_signal_v2,
        "signal_v3": rp_factor_signal_v3,
    }


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


def _lead_lag_virtual_returns(
    returns: pd.DataFrame,
    lagged_signal: pd.DataFrame,
) -> pd.DataFrame:
    columns: dict[tuple[str, str], pd.Series] = {}
    for asset in returns.columns:
        for signal_asset in lagged_signal.columns:
            columns[(str(asset), str(signal_asset))] = returns[asset] * lagged_signal[signal_asset]
    virtual = pd.DataFrame(columns, index=returns.index, dtype=float)
    virtual.columns = pd.MultiIndex.from_tuples(virtual.columns, names=["asset", "signal_asset"])
    return virtual


def _lead_lag_symmetric_virtual_returns(
    returns: pd.DataFrame,
    lagged_signal: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    tickers = list(returns.columns)
    pairs: list[tuple[int, int]] = []
    columns: dict[tuple[str, str], pd.Series] = {}
    for i, asset_i in enumerate(tickers):
        for j in range(i, len(tickers)):
            asset_j = tickers[j]
            key = (str(asset_i), str(asset_j))
            if i == j:
                columns[key] = returns[asset_i] * lagged_signal[asset_i]
            else:
                columns[key] = (returns[asset_i] * lagged_signal[asset_j]) + (returns[asset_j] * lagged_signal[asset_i])
            pairs.append((i, j))
    virtual = pd.DataFrame(columns, index=returns.index, dtype=float)
    virtual.columns = pd.MultiIndex.from_tuples(virtual.columns, names=["asset", "signal_asset"])
    return virtual, pairs


def _lead_lag_trend_following_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    long_only: bool,
    target_dates: pd.Index | None = None,
) -> pd.DataFrame:
    _, _, z_returns = _sanitized_normalized_returns(prices, est_cfg)
    signal = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    lagged_signal = signal.shift(1)
    virtual_returns, virtual_pairs = _lead_lag_symmetric_virtual_returns(z_returns, lagged_signal)
    covariance_alpha = est_cfg.covariance_alpha
    if covariance_alpha is None:
        covariance_alpha = 2.0 / (float(est_cfg.corr_span) + 1.0)
    alpha = float(covariance_alpha)
    target_dates = prices.index if target_dates is None else pd.Index(target_dates)
    target_set = set(pd.DatetimeIndex(target_dates))
    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex(target_dates), columns=prices.columns, dtype=float)
    regularization = max(float(est_cfg.lltf_l2_reg), 0.0)
    num_virtual = len(virtual_pairs)
    mean_vec = np.zeros(num_virtual, dtype=float)
    second_moment = np.zeros((num_virtual, num_virtual), dtype=float)
    seen = 0
    num_assets = len(prices.columns)

    for ts in prices.index:
        x = virtual_returns.loc[ts].fillna(0.0).to_numpy(dtype=float)
        if np.any(np.isfinite(x)):
            seen += 1
            mean_vec = ((1.0 - alpha) * mean_vec) + (alpha * x)
            second_moment = ((1.0 - alpha) * second_moment) + (alpha * np.outer(x, x))
        if ts not in target_set or seen < est_cfg.covariance_min_periods:
            continue

        cov_arr = second_moment - np.outer(mean_vec, mean_vec)
        cov_arr = cov_arr + regularization * np.eye(num_virtual, dtype=float)
        beta = np.linalg.pinv(cov_arr) @ mean_vec
        omega_matrix = np.zeros((num_assets, num_assets), dtype=float)
        for coeff, (i, j) in zip(beta, virtual_pairs):
            omega_matrix[i, j] = coeff
            omega_matrix[j, i] = coeff
        signal_vec = signal.loc[ts].reindex(prices.columns).fillna(0.0).to_numpy(dtype=float)
        raw = pd.Series(omega_matrix @ signal_vec, index=prices.columns, dtype=float)
        weights.loc[ts] = normalize_weights(raw, long_only=long_only)

    return weights.ffill().fillna(0.0)


def _compute_base_weights_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.Series:
    if strategy == "EW":
        available = prices.loc[date].dropna().index
        if len(available) == 0:
            return pd.Series(0.0, index=prices.columns, dtype=float)
        row = pd.Series(0.0, index=prices.columns, dtype=float)
        row.loc[available] = 1.0 / len(available)
        return normalize_weights(row, long_only=long_only)
    cov = _resolve_covariance_at_date(prices, est_cfg, date, covariance_cache)
    if strategy in strategy_registry():
        raw = resolve_strategy(strategy)(cov).reindex(prices.columns).fillna(0.0)
        if long_only:
            raw = normalize_weights(raw, long_only=True)
        return raw.astype(float)
    raise KeyError(f"Unsupported base strategy '{strategy}'.")


def _trend_on_risk_parity_v0_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyState:
    history = prices.loc[prices.index <= date]
    _, _, z_returns = _sanitized_normalized_returns(history, est_cfg)
    trend = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    cov = _resolve_covariance_at_date(history, est_cfg, date, covariance_cache)
    base = risk_parity_weights_from_cov(cov).reindex(prices.columns).fillna(0.0)
    signal = trend.loc[date].reindex(base.index).fillna(0.0)
    effective = normalize_weights(base * signal, long_only=long_only).reindex(prices.columns).fillna(0.0)
    return StrategyState(base_weights=base, signal_scale=1.0, effective_weights=effective)


def _trend_on_risk_parity_v1_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyState:
    history = prices.loc[prices.index <= date]
    _, _, z_returns = _sanitized_normalized_returns(history, est_cfg)
    trend = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    cov = _resolve_covariance_at_date(history, est_cfg, date, covariance_cache)
    base = risk_parity_weights_from_cov(cov).reindex(prices.columns).fillna(0.0)
    effective = trend_on_risk_parity_weights_from_cov_and_signal(
        cov,
        trend.loc[date],
        long_only=long_only,
    ).reindex(prices.columns).fillna(0.0)
    projected_signal = 0.0 if base.abs().sum() == 0 else float(effective @ base)
    return StrategyState(base_weights=base, signal_scale=projected_signal, effective_weights=effective)


def _trend_on_risk_parity_v2_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    torp_context: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> StrategyState:
    tilt = _torp_rp_tilt(prices.columns)
    history = prices.loc[prices.index <= date]
    if torp_context is None:
        cov_panel = _resolve_covariance_cache_until_date(history, est_cfg, date, covariance_cache)
        torp_context = _build_torp_factor_context(history, est_cfg, cov_panel)
    cov = _resolve_covariance_at_date(history, est_cfg, date, covariance_cache)
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


def _trend_on_risk_parity_v3_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    torp_context: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> StrategyState:
    history = prices.loc[prices.index <= date]
    if torp_context is None:
        cov_panel = _resolve_covariance_cache_until_date(history, est_cfg, date, covariance_cache)
        torp_context = _build_torp_factor_context(history, est_cfg, cov_panel)
    base_weights = torp_context["base_weights"]
    raw_signal = torp_context["signal_v3"].loc[date]
    factor_signal = 0.0 if pd.isna(raw_signal) else float(raw_signal)
    factor_signal *= float(est_cfg.torp_signal_gain)
    base = base_weights.loc[date].reindex(prices.columns).fillna(0.0)
    effective = base * factor_signal
    if long_only:
        effective = effective.clip(lower=0.0)
    return StrategyState(base_weights=base, signal_scale=factor_signal, effective_weights=effective)


def _lead_lag_trend_following_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
) -> StrategyState:
    weights = _lead_lag_trend_following_panel(
        prices.loc[prices.index <= date],
        est_cfg,
        long_only=long_only,
        target_dates=pd.DatetimeIndex([date]),
    ).loc[date].reindex(prices.columns).fillna(0.0)
    return _weights_to_strategy_state(weights)


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
    if strategy == "ToRP0":
        return _trend_on_risk_parity_v0_at_date(prices, est_cfg, date=ts, long_only=long_only, covariance_cache=covariance_cache)
    if strategy == "ToRP1":
        return _trend_on_risk_parity_v1_at_date(prices, est_cfg, date=ts, long_only=long_only, covariance_cache=covariance_cache)
    if strategy == "ToRP2":
        return _trend_on_risk_parity_v2_at_date(
            prices,
            est_cfg,
            date=ts,
            long_only=long_only,
            covariance_cache=covariance_cache,
            torp_context=strategy_context,
        )
    if strategy == "ToRP3":
        return _trend_on_risk_parity_v3_at_date(
            prices,
            est_cfg,
            date=ts,
            long_only=long_only,
            covariance_cache=covariance_cache,
            torp_context=strategy_context,
        )
    if strategy == "LLTF":
        return _lead_lag_trend_following_at_date(prices, est_cfg, date=ts, long_only=long_only)
    return _weights_to_strategy_state(
        _compute_base_weights_at_date(prices, est_cfg, strategy, date=ts, long_only=long_only, covariance_cache=covariance_cache)
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
        strategy_context = _build_torp_factor_context(prices, est_cfg, covariance_cache)
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
