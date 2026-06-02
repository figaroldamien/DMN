from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Callable

import pandas as pd

from trading_core.reporting import EvaluationSummary, cumulative_nav

from .types import EvaluationResult


@dataclass(frozen=True)
class ComparisonResult:
    strategy_results: dict[str, EvaluationResult]
    summary_table: pd.DataFrame
    nav_comparison: pd.DataFrame
    drawdown_comparison: pd.DataFrame


def _empty_summary_columns() -> list[str]:
    return [
        "strategy",
        *asdict(
            EvaluationSummary(
                total_return=0.0,
                ann_return=0.0,
                ann_vol=0.0,
                sharpe=0.0,
                mdd=0.0,
                avg_turnover=0.0,
                annualized_turnover=0.0,
                total_cost=0.0,
                annualized_cost=0.0,
                pct_positive_days=0.0,
                num_days=0,
                num_rebalances=0,
            )
        ).keys(),
    ]


def evaluate_strategies(
    prices: pd.DataFrame,
    estimation: Any,
    backtest: Any,
    evaluation: Any,
    strategies: list[str],
    *,
    evaluate_portfolio_fn: Callable[[pd.DataFrame, Any, Any, Any], EvaluationResult],
) -> dict[str, EvaluationResult]:
    out: dict[str, EvaluationResult] = {}
    for strategy in strategies:
        if is_dataclass(evaluation):
            eval_cfg = replace(evaluation, strategy=strategy)
        else:
            eval_cfg = copy.copy(evaluation)
            eval_cfg.strategy = strategy
        out[strategy] = evaluate_portfolio_fn(prices, estimation, backtest, eval_cfg)
    return out


def build_summary_table(strategy_results: dict[str, EvaluationResult]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for strategy, result in strategy_results.items():
        payload = asdict(result.summary)
        payload["strategy"] = strategy
        rows.append(payload)
    if not rows:
        return pd.DataFrame(columns=_empty_summary_columns())
    table = pd.DataFrame(rows)
    cols = ["strategy", *[col for col in table.columns if col != "strategy"]]
    return table.loc[:, cols].sort_values("sharpe", ascending=False).reset_index(drop=True)


def build_nav_comparison(strategy_results: dict[str, EvaluationResult]) -> pd.DataFrame:
    series = {
        strategy: cumulative_nav(result.daily_returns_net).rename(strategy)
        for strategy, result in strategy_results.items()
    }
    if not series:
        return pd.DataFrame()
    return pd.concat(series.values(), axis=1).sort_index().ffill()


def build_drawdown_comparison(nav_comparison: pd.DataFrame) -> pd.DataFrame:
    if nav_comparison.empty:
        return pd.DataFrame()
    return nav_comparison.divide(nav_comparison.cummax()).subtract(1.0)


def compare_strategies(
    prices: pd.DataFrame,
    estimation: Any,
    backtest: Any,
    evaluation: Any,
    strategies: list[str],
    *,
    evaluate_portfolio_fn: Callable[[pd.DataFrame, Any, Any, Any], EvaluationResult],
) -> ComparisonResult:
    strategy_results = evaluate_strategies(
        prices,
        estimation,
        backtest,
        evaluation,
        strategies,
        evaluate_portfolio_fn=evaluate_portfolio_fn,
    )
    nav_comparison = build_nav_comparison(strategy_results)
    return ComparisonResult(
        strategy_results=strategy_results,
        summary_table=build_summary_table(strategy_results),
        nav_comparison=nav_comparison,
        drawdown_comparison=build_drawdown_comparison(nav_comparison),
    )
