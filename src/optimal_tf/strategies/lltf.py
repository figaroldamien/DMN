from __future__ import annotations

"""Lead-Lag Trend Following strategy.

LLTF is the most structurally different strategy in the current codebase.
Unlike RP / ARP / NM, it is not a direct mapping from a covariance matrix to a
static weight vector. Instead, it tries to learn directional interactions across
assets from lagged trend information.

Core intuition:
- each asset has a current return ``r_j``
- each asset also has a lagged trend signal ``s_k``
- LLTF studies products of the form ``r_j * s_k``
- if these cross terms have persistent structure, they indicate which assets
  tend to respond when trend signals appear elsewhere in the universe

The implementation below keeps the math explicit so the strategy can be audited
and modified without treating it as a black box.
"""

import numpy as np
import pandas as pd

from ..config import EstimationConfig
from ..features import alpha_from_span, effective_span_from_alpha, trend_ema_signal
from .weights import normalize_weights
from .common import sanitized_normalized_returns, weights_to_strategy_state
from .types import StrategyState


def _lead_lag_virtual_returns(
    returns: pd.DataFrame,
    lagged_signal: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full lead-lag virtual return panel.

    For every ordered pair ``(j, k)``, we create the series:

        r_j(t) * s_k(t-1)

    Interpretation:
    - the first index identifies the asset whose return is being explained
    - the second identifies the asset whose lagged trend is used as a signal

    This full cross-product version is pedagogically useful because it exposes
    the original LLTF construction directly, even though the production code
    later switches to a symmetric compressed representation.
    """
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
    """Build a symmetric compressed virtual-return representation.

    The unconstrained LLTF interaction matrix would contain one coefficient for
    every ordered pair of assets. Here we impose symmetry, which means that the
    pair ``(i, j)`` and ``(j, i)`` share the same parameter.

    Consequences:
    - the parameter count is reduced from ``N^2`` to ``N(N+1)/2``
    - estimation is more stable
    - the resulting interaction matrix is easier to interpret

    For off-diagonal pairs, we explicitly fold both directions together.
    """
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
    """Compute LLTF weights on a panel of rebalance dates.

    Pipeline overview:
    1. sanitize prices into volatility-normalized returns
    2. build lagged trend signals
    3. construct virtual returns ``r_j * s_k``
    4. maintain EWMA estimates of their mean and covariance
    5. solve a regularized linear system for interaction coefficients
    6. turn those coefficients back into a current cross-sectional weight vector

    This is effectively a dynamic mean-variance problem in the space of
    interaction terms rather than in the space of raw assets.
    """
    _, _, z_returns = sanitized_normalized_returns(prices, est_cfg)
    signal = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    lagged_signal = signal.shift(1)
    virtual_returns, virtual_pairs = _lead_lag_symmetric_virtual_returns(z_returns, lagged_signal)

    # The LLTF optimizer uses an EWMA estimation scheme. We try the dedicated
    # covariance alpha first and fall back to a span-derived value if needed.
    covariance_alpha = est_cfg.covariance_alpha
    if covariance_alpha is None:
        covariance_alpha = alpha_from_span(est_cfg.covariance_window)
    if covariance_alpha is None:
        covariance_alpha = alpha_from_span(est_cfg.corr_span)
    alpha = float(covariance_alpha)

    target_dates = prices.index if target_dates is None else pd.Index(target_dates)
    target_set = set(pd.DatetimeIndex(target_dates))
    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex(target_dates), columns=prices.columns, dtype=float)

    # L2 regularization stabilizes the inverse problem on the virtual asset
    # covariance matrix, which can otherwise be very noisy or nearly singular.
    regularization = max(float(est_cfg.lltf_l2_reg), 0.0)
    num_virtual = len(virtual_pairs)
    mean_vec = np.zeros(num_virtual, dtype=float)
    second_moment = np.zeros((num_virtual, num_virtual), dtype=float)
    seen = 0
    num_assets = len(prices.columns)

    # If no explicit minimum is given, derive a rough warm-up period from the
    # EWMA effective span so the optimization does not start too early.
    min_periods = est_cfg.covariance_min_periods
    if min_periods <= 0:
        min_periods = effective_span_from_alpha(alpha) or 1

    for ts in prices.index:
        # Missing virtual returns are replaced with zero here so the EWMA state
        # continues to evolve smoothly without exploding on sparse observations.
        x = virtual_returns.loc[ts].fillna(0.0).to_numpy(dtype=float)
        if np.any(np.isfinite(x)):
            seen += 1
            mean_vec = ((1.0 - alpha) * mean_vec) + (alpha * x)
            second_moment = ((1.0 - alpha) * second_moment) + (alpha * np.outer(x, x))

        if ts not in target_set or seen < min_periods:
            continue

        # Covariance estimate of the virtual-return process.
        cov_arr = second_moment - np.outer(mean_vec, mean_vec)
        cov_arr = cov_arr + regularization * np.eye(num_virtual, dtype=float)

        # ``beta`` is the vector of optimal interaction weights in virtual-asset
        # space under the regularized mean-variance approximation.
        beta = np.linalg.pinv(cov_arr) @ mean_vec

        # Rebuild the symmetric interaction matrix Omega from the compressed
        # upper-triangle representation.
        omega_matrix = np.zeros((num_assets, num_assets), dtype=float)
        for coeff, (i, j) in zip(beta, virtual_pairs):
            omega_matrix[i, j] = coeff
            omega_matrix[j, i] = coeff

        # Apply the learned interaction structure to today's trend vector. This
        # maps the model back from factor/interactions space into asset weights.
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
    """Compute LLTF on a single allocation date.

    The date-level API delegates to the panel engine on a singleton target date.
    This keeps the core optimization logic in one place and avoids maintaining a
    separate one-date implementation that could drift from the panel version.
    """
    weights = lead_lag_trend_following_panel(
        prices.loc[prices.index <= date],
        est_cfg,
        long_only=long_only,
        target_dates=pd.DatetimeIndex([date]),
    ).loc[date].reindex(prices.columns).fillna(0.0)
    return weights_to_strategy_state(weights)
