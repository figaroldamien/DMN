"""Shared periodic backtest engine primitives."""

from .comparison import ComparisonResult, compare_strategies
from .engine import apply_portfolio_vol_target, evaluate_portfolio, slice_next_holding_period
from .types import EvaluationResult

__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "apply_portfolio_vol_target",
    "compare_strategies",
    "evaluate_portfolio",
    "slice_next_holding_period",
]
