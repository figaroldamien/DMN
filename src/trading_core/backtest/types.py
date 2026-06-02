from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_core.reporting.metrics import EvaluationSummary


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
