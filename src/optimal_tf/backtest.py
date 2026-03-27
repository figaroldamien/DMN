from __future__ import annotations

import math
from typing import Callable

import pandas as pd

from .config import BacktestConfig, EstimationConfig
from .estimators.pipeline import estimate_clean_covariance_panel
from .features import compute_returns


def _portfolio_returns_from_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[pd.Series, pd.Series]:
    sigma_daily = cfg.sigma_target_annual / math.sqrt(252)
    # Positions are lagged by one bar to avoid look-ahead bias between
    # estimation/allocation and realized returns.
    gross = (weights.shift(1) * returns).sum(axis=1)
    if cfg.portfolio_vol_target:
        realized = gross.ewm(span=cfg.portfolio_vol_span, adjust=False, min_periods=cfg.portfolio_vol_span).std()
        scale = (sigma_daily / (realized + 1e-12)).clip(0.0, 5.0)
        gross = gross * scale
    turnover = weights.diff().abs().sum(axis=1)
    net = gross - (cfg.cost_bps / 1e4) * turnover
    return net, turnover


def build_weight_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    weight_fn: Callable[[pd.DataFrame], pd.Series],
    *,
    long_only: bool = False,
) -> pd.DataFrame:
    cov_panel = estimate_clean_covariance_panel(prices, est_cfg)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ts, cov in cov_panel.items():
        w = weight_fn(cov).reindex(prices.columns).fillna(0.0)
        # Long-only is applied after the portfolio recipe so the same weight
        # builder can be reused for both constrained and unconstrained tests.
        if long_only:
            w = w.clip(lower=0.0)
            if w.sum() > 0:
                w = w / w.sum()
        weights.loc[ts] = w
    return weights.ffill().fillna(0.0)


def backtest_portfolio(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    bt_cfg: BacktestConfig,
    weight_fn: Callable[[pd.DataFrame], pd.Series],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    returns = compute_returns(prices)
    weights = build_weight_panel(prices, est_cfg, weight_fn, long_only=bt_cfg.long_only)
    pnl, turnover = _portfolio_returns_from_weights(returns, weights, bt_cfg)
    return pnl, turnover, weights
