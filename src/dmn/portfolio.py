from __future__ import annotations

import math
from typing import Tuple

import pandas as pd

from .config import BacktestConfig
from .features import compute_returns, ewma_vol


def compute_turnover(x: pd.DataFrame, daily_vol: pd.DataFrame, sigma_target_annual: float) -> pd.Series:
    sigma_tgt_daily = sigma_target_annual / math.sqrt(252)
    w = (x / (daily_vol + 1e-12)) * sigma_tgt_daily
    return w.diff().abs().mean(axis=1)


def run_portfolio(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    rets = compute_returns(prices)
    daily_vol = ewma_vol(rets, span=cfg.vol_span)
    sigma_tgt_daily = cfg.sigma_target_annual / math.sqrt(252)

    scaled_weights = positions * (sigma_tgt_daily / (daily_vol + 1e-12))
    strat = (scaled_weights.shift(1) * rets).mean(axis=1)

    turnover = compute_turnover(positions, daily_vol, cfg.sigma_target_annual)
    cost = (cfg.cost_bps / 1e4) * turnover
    strat_ex_cost = strat - cost

    if cfg.portfolio_vol_target:
        vol_port = strat_ex_cost.ewm(span=60, adjust=False, min_periods=60).std()
        scale = (sigma_tgt_daily / (vol_port + 1e-12)).clip(0.0, 5.0)
        strat_ex_cost = strat_ex_cost * scale

    return strat_ex_cost, turnover, scaled_weights
