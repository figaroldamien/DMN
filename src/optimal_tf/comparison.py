from trading_core.backtest.comparison import (
    ComparisonResult,
    build_drawdown_comparison,
    build_nav_comparison,
    build_summary_table,
    compare_strategies as _core_compare_strategies,
    evaluate_strategies as _core_evaluate_strategies,
)
from trading_core.backtest.types import EvaluationResult

from .evaluation import evaluate_portfolio


def evaluate_strategies(
    prices,
    estimation,
    backtest,
    evaluation,
    strategies,
) -> dict[str, EvaluationResult]:
    return _core_evaluate_strategies(
        prices,
        estimation,
        backtest,
        evaluation,
        strategies,
        evaluate_portfolio_fn=evaluate_portfolio,
    )


def compare_strategies(
    prices,
    estimation,
    backtest,
    evaluation,
    strategies,
) -> ComparisonResult:
    return _core_compare_strategies(
        prices,
        estimation,
        backtest,
        evaluation,
        strategies,
        evaluate_portfolio_fn=evaluate_portfolio,
    )


__all__ = [
    "ComparisonResult",
    "build_drawdown_comparison",
    "build_nav_comparison",
    "build_summary_table",
    "compare_strategies",
    "evaluate_strategies",
]
