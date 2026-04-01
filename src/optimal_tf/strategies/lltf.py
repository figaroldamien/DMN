from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EstimationConfig
from ..features import alpha_from_span, effective_span_from_alpha, trend_ema_signal
from ..portfolios import normalize_weights
from .common import sanitized_normalized_returns, weights_to_strategy_state
from .types import StrategyState


def _lead_lag_virtual_returns(
    returns: pd.DataFrame,
    lagged_signal: pd.DataFrame,
) -> pd.DataFrame:
    # This full cross-product version is kept because it is the easiest way to
    # understand the LLTF construction conceptually, even though the production
    # path uses the symmetric variant below.
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
    # The LLTF optimizer works on "virtual assets" r_j * s_k. Because the
    # interaction matrix is constrained to be symmetric here, we only keep the
    # upper triangle and fold off-diagonal terms together.
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


def lead_lag_trend_following_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    long_only: bool,
    target_dates: pd.Index | None = None,
) -> pd.DataFrame:
    # LLTF estimates a mean-variance problem on cross-asset trend interactions:
    # 1. build lagged trend signals per asset
    # 2. form virtual returns r_j * s_k
    # 3. maintain EWMA first/second moments on those virtual assets
    # 4. solve a regularized linear system for interaction weights
    # 5. map the interaction matrix back to current asset exposures
    _, _, z_returns = sanitized_normalized_returns(prices, est_cfg)
    signal = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    lagged_signal = signal.shift(1)
    virtual_returns, virtual_pairs = _lead_lag_symmetric_virtual_returns(z_returns, lagged_signal)
    covariance_alpha = est_cfg.covariance_alpha
    if covariance_alpha is None:
        covariance_alpha = alpha_from_span(est_cfg.covariance_window)
    if covariance_alpha is None:
        covariance_alpha = alpha_from_span(est_cfg.corr_span)
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
    min_periods = est_cfg.covariance_min_periods
    if min_periods <= 0:
        min_periods = effective_span_from_alpha(alpha) or 1

    for ts in prices.index:
        x = virtual_returns.loc[ts].fillna(0.0).to_numpy(dtype=float)
        if np.any(np.isfinite(x)):
            seen += 1
            mean_vec = ((1.0 - alpha) * mean_vec) + (alpha * x)
            second_moment = ((1.0 - alpha) * second_moment) + (alpha * np.outer(x, x))
        if ts not in target_set or seen < min_periods:
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


def lead_lag_trend_following_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp,
    long_only: bool,
) -> StrategyState:
    # The date-level API delegates to the panel engine on a singleton target
    # date so the optimization logic stays in one place.
    weights = lead_lag_trend_following_panel(
        prices.loc[prices.index <= date],
        est_cfg,
        long_only=long_only,
        target_dates=pd.DatetimeIndex([date]),
    ).loc[date].reindex(prices.columns).fillna(0.0)
    return weights_to_strategy_state(weights)
