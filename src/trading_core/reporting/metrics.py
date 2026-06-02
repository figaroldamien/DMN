from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Performance:
    ann_return: float
    ann_vol: float
    sharpe: float
    mdd: float
    avg_turnover: float


@dataclass(frozen=True)
class EvaluationSummary:
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    mdd: float
    avg_turnover: float
    annualized_turnover: float
    total_cost: float
    annualized_cost: float
    pct_positive_days: float
    num_days: int
    num_rebalances: int


def performance_metrics(pnl: pd.Series, turnover: pd.Series) -> Performance:
    pnl = pnl.dropna()
    ann_return = pnl.mean() * 252
    ann_vol = pnl.std() * math.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    # We build drawdown from the compounded path so the metric is aligned with
    # what we will later inspect in portfolio equity curves.
    wealth = (1.0 + pnl.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return Performance(
        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        mdd=float(drawdown.min()),
        avg_turnover=float(turnover.mean()),
    )


def evaluation_metrics(
    pnl: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    *,
    num_rebalances: int | None = None,
) -> EvaluationSummary:
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return EvaluationSummary(
            total_return=0.0,
            ann_return=0.0,
            ann_vol=0.0,
            sharpe=0.0,
            mdd=0.0,
            avg_turnover=0.0,
            annualized_turnover=0.0,
            total_cost=float(costs.sum()),
            annualized_cost=0.0,
            pct_positive_days=0.0,
            num_days=0,
            num_rebalances=int(num_rebalances or 0),
        )
    ann_return = pnl.mean() * 252
    ann_vol = pnl.std() * math.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    wealth = (1.0 + pnl).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    total_return = float(wealth.iloc[-1] - 1.0)
    total_cost = float(costs.sum())
    years = max(len(pnl) / 252.0, 1.0 / 252.0)
    return EvaluationSummary(
        total_return=total_return,
        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        mdd=float(drawdown.min()),
        avg_turnover=float(turnover.mean()),
        annualized_turnover=float(turnover.mean() * 252),
        total_cost=total_cost,
        annualized_cost=float(total_cost / years),
        pct_positive_days=float((pnl > 0).mean()),
        num_days=int(len(pnl)),
        num_rebalances=int(num_rebalances if num_rebalances is not None else len(costs)),
    )
