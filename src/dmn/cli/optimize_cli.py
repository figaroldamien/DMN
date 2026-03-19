from __future__ import annotations

import argparse
from typing import Sequence

from ..data import load_prices_yf
from ..metrics import performance_metrics
from ..optimize import (
    optimization_config_to_dict,
    optimization_summary,
    run_grid_search,
    validate_optimization_config,
)
from ..portfolio import run_portfolio
from ..strategies import strategy_long_only
from .common import (
    add_strategy_choice_arg,
    build_parser_with_argsets,
    format_results_table,
    infer_strategy_argsets,
    load_effective_run_config,
    print_config_payload,
    resolve_config_tickers,
    strategy_choice_names,
)


def build_parser() -> argparse.ArgumentParser:
    strategy_choices = strategy_choice_names(supports_optimization=True)
    parser = build_parser_with_argsets(
        "Run a grid search on one sequence strategy.",
        "config",
        "universe",
        "backtest",
        *infer_strategy_argsets(strategy_choices, allowed_argsets={"optimization"}),
    )

    add_strategy_choice_arg(
        parser,
        supports_optimization=True,
        help_text="Strategy to optimize.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_effective_run_config(args, include_optimization=True)

    if cfg.optimization is None:
        parser.error("Optimization config is required. Provide an [optimization] section or CLI overrides.")
    try:
        validate_optimization_config(cfg.optimization)
    except ValueError as exc:
        parser.error(str(exc))

    tickers = resolve_config_tickers(cfg, parser)

    if args.print_config:
        payload = cfg.to_dict()
        payload["optimization"] = optimization_config_to_dict(cfg.optimization)
        print_config_payload(payload)

    try:
        prices = load_prices_yf(tickers, start=cfg.start)
    except ImportError:
        print("Install yfinance to run the example: pip install yfinance")
        return 0

    long_only_positions = strategy_long_only(prices)
    long_only_returns, long_only_turnover, _ = run_portfolio(prices, long_only_positions, cfg.backtest)
    long_only_perf = performance_metrics(long_only_returns, long_only_turnover)

    results = run_grid_search(prices, cfg.backtest, cfg.optimization)
    best = optimization_summary(results)

    display_res = format_results_table(results, width=180, drop_columns=["strategy"])
    long_only_row = {
        "ann_return": long_only_perf.ann_return,
        "ann_vol": long_only_perf.ann_vol,
        "sharpe": long_only_perf.sharpe,
        "sortino": long_only_perf.sortino,
        "calmar": long_only_perf.calmar,
        "mdd": long_only_perf.mdd,
        "pct_pos": long_only_perf.pct_pos,
        "avgP_over_avgL": long_only_perf.avg_profit_over_avg_loss,
        "avg_turnover": long_only_perf.avg_turnover,
    }
    long_only_summary = ", ".join(
        f"{column}={value:.3f}" for column, value in long_only_row.items() if column in display_res.columns
    )
    print(f"LongOnly: {long_only_summary}")
    print(display_res.to_string(index=False, col_space=5))
    print()
    print("Best candidate:")
    print_config_payload(best, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
