from __future__ import annotations

from dataclasses import dataclass, field

import math
import pandas as pd

from .allocation import compute_strategy_panel
from .config import BacktestConfig, EstimationConfig, EvaluationConfig
from .estimators.pipeline import estimate_clean_covariance_panel
from .features import compute_returns, sanitize_returns
from .metrics import EvaluationSummary, evaluation_metrics
from .rebalance import resolve_rebalance_dates


@dataclass(frozen=True)
class EvaluationResult:
    summary: EvaluationSummary
    weights_by_rebalance: pd.DataFrame
    daily_returns_gross: pd.Series
    daily_returns_net: pd.Series
    turnover_by_rebalance: pd.Series
    costs_by_rebalance: pd.Series
    holding_period_returns_gross: pd.Series
    holding_period_returns_net: pd.Series
    base_weights_by_rebalance: pd.DataFrame = field(default_factory=pd.DataFrame)
    effective_weights_by_rebalance: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_scale_by_rebalance: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    portfolio_vol_scale: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _apply_portfolio_vol_target(
    gross_returns: pd.Series,
    bt_cfg: BacktestConfig,
) -> tuple[pd.Series, pd.Series]:
    if not bt_cfg.portfolio_vol_target:
        return gross_returns, pd.Series(1.0, index=gross_returns.index, dtype=float)
    sigma_daily = bt_cfg.sigma_target_annual / math.sqrt(252)
    realized = gross_returns.ewm(
        span=bt_cfg.portfolio_vol_span,
        adjust=False,
        min_periods=bt_cfg.portfolio_vol_span,
    ).std()
    scale = (sigma_daily / (realized + 1e-12)).clip(0.0, 5.0).shift(1).fillna(1.0)
    return gross_returns * scale, scale


def _slice_next_holding_period(
    index: pd.DatetimeIndex,
    current: pd.Timestamp,
    next_rebalance: pd.Timestamp | None,
    evaluation_end: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    mask = index > current
    if next_rebalance is not None:
        mask &= index <= next_rebalance
    if evaluation_end is not None:
        mask &= index <= evaluation_end
    return index[mask]


def evaluate_portfolio(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    bt_cfg: BacktestConfig,
    eval_cfg: EvaluationConfig,
) -> EvaluationResult:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=est_cfg.max_abs_return)
    rebalance_dates = resolve_rebalance_dates(
        prices.index,
        eval_cfg.rebalance_frequency,
        start=eval_cfg.evaluation_start,
        end=eval_cfg.evaluation_end,
    )
    covariance_cache = estimate_clean_covariance_panel(prices, est_cfg)
    strategy_panel = compute_strategy_panel(
        prices,
        est_cfg,
        eval_cfg.strategy,
        long_only=bt_cfg.long_only,
        target_dates=rebalance_dates,
        covariance_cache=covariance_cache,
    )
    base_weights_by_rebalance = strategy_panel.base_weights.loc[rebalance_dates].copy()
    effective_weights_by_rebalance = strategy_panel.effective_weights.loc[rebalance_dates].copy()
    signal_scale_by_rebalance = strategy_panel.signal_scale.loc[rebalance_dates].copy()
    weights_by_rebalance = effective_weights_by_rebalance.copy()

    daily_gross = pd.Series(0.0, index=prices.index, dtype=float)
    daily_net = pd.Series(0.0, index=prices.index, dtype=float)
    turnover = pd.Series(0.0, index=rebalance_dates, dtype=float)
    costs = pd.Series(0.0, index=rebalance_dates, dtype=float)
    prev_weights = pd.Series(0.0, index=prices.columns, dtype=float)

    eval_end = pd.Timestamp(eval_cfg.evaluation_end) if eval_cfg.evaluation_end is not None else None

    for pos, rebalance_date in enumerate(rebalance_dates):
        current_weights = effective_weights_by_rebalance.loc[rebalance_date].fillna(0.0)
        next_rebalance = rebalance_dates[pos + 1] if pos + 1 < len(rebalance_dates) else None
        period_index = _slice_next_holding_period(prices.index, rebalance_date, next_rebalance, eval_end)
        if len(period_index) == 0:
            continue

        rebalance_turnover = float((current_weights - prev_weights).abs().sum())
        rebalance_cost = (bt_cfg.cost_bps / 1e4) * rebalance_turnover
        turnover.loc[rebalance_date] = rebalance_turnover
        costs.loc[rebalance_date] = rebalance_cost

        gross_slice = (returns.loc[period_index].fillna(0.0) * current_weights).sum(axis=1)
        daily_gross.loc[period_index] = gross_slice
        prev_weights = current_weights

    daily_gross, portfolio_vol_scale = _apply_portfolio_vol_target(daily_gross, bt_cfg)
    daily_net = daily_gross.copy()
    for rebalance_date, rebalance_cost in costs.items():
        if rebalance_cost == 0.0:
            continue
        next_period = daily_net.index[daily_net.index > rebalance_date]
        if len(next_period) == 0:
            continue
        daily_net.loc[next_period[0]] -= rebalance_cost

    if eval_cfg.evaluation_start is not None:
        start_ts = pd.Timestamp(eval_cfg.evaluation_start)
        daily_gross = daily_gross.loc[daily_gross.index >= start_ts]
        daily_net = daily_net.loc[daily_net.index >= start_ts]
        portfolio_vol_scale = portfolio_vol_scale.loc[portfolio_vol_scale.index >= start_ts]
    if eval_end is not None:
        daily_gross = daily_gross.loc[daily_gross.index <= eval_end]
        daily_net = daily_net.loc[daily_net.index <= eval_end]
        portfolio_vol_scale = portfolio_vol_scale.loc[portfolio_vol_scale.index <= eval_end]

    holding_gross = pd.Series(index=rebalance_dates, dtype=float)
    holding_net = pd.Series(index=rebalance_dates, dtype=float)
    for pos, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[pos + 1] if pos + 1 < len(rebalance_dates) else None
        period_index = _slice_next_holding_period(prices.index, rebalance_date, next_rebalance, eval_end)
        if len(period_index) == 0:
            continue
        holding_gross.loc[rebalance_date] = float((1.0 + daily_gross.loc[period_index]).prod() - 1.0)
        holding_net.loc[rebalance_date] = float((1.0 + daily_net.loc[period_index]).prod() - 1.0)

    summary = evaluation_metrics(
        daily_net,
        turnover.reindex(daily_net.index).fillna(0.0),
        costs,
        num_rebalances=int(len(holding_net.dropna())),
    )
    return EvaluationResult(
        summary=summary,
        weights_by_rebalance=weights_by_rebalance,
        daily_returns_gross=daily_gross,
        daily_returns_net=daily_net,
        turnover_by_rebalance=turnover,
        costs_by_rebalance=costs,
        holding_period_returns_gross=holding_gross.dropna(),
        holding_period_returns_net=holding_net.dropna(),
        base_weights_by_rebalance=base_weights_by_rebalance,
        effective_weights_by_rebalance=effective_weights_by_rebalance,
        signal_scale_by_rebalance=signal_scale_by_rebalance,
        portfolio_vol_scale=portfolio_vol_scale,
    )
