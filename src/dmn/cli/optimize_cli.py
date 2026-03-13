from __future__ import annotations

import argparse
import json
from typing import Sequence

import pandas as pd

from market_tickers import MARKET_TICKERS

from ..config import RunConfig
from ..config_io import load_run_config, merge_cli_overrides, merge_optimization_cli_overrides
from ..data import load_prices_yf
from ..metrics import performance_metrics
from ..optimize import (
    optimization_config_to_dict,
    optimization_summary,
    run_grid_search,
    strategy_registry,
    validate_optimization_config,
)
from ..portfolio import run_portfolio
from ..strategies import strategy_long_only
from ..universe import resolve_tickers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a grid search on one sequence strategy.")
    parser.add_argument("--config", default=None, help="Optional path to a TOML/JSON configuration file.")
    parser.add_argument("--no-print-config", dest="print_config", action="store_false", default=True, help="Disable effective configuration display.")
    parser.add_argument("--market", default=None, choices=sorted(MARKET_TICKERS.keys()), help="Ticker universe to load.")
    parser.add_argument("--ticker", default=None, help="Single ticker to load directly (exclusive with --market).")
    parser.add_argument("--start", default=None, help="Start date for yfinance download (YYYY-MM-DD).")
    parser.add_argument("--sector", default=None, help="Optional sector filter (for component-based indices).")
    parser.add_argument("--sub-sector", dest="sub_sector", default=None, help="Optional sub-sector filter (for component-based indices).")

    parser.add_argument("--sigma-target-annual", type=float, default=None, help="Target annualized volatility.")
    parser.add_argument("--vol-span", type=int, default=None, help="EWMA span for daily volatility.")
    parser.add_argument("--cost-bps", type=float, default=None, help="Transaction cost in bps per turnover unit.")
    parser.add_argument("--min-obs", type=int, default=None, help="Minimum warmup observations.")
    parser.add_argument(
        "--portfolio-vol-target",
        dest="portfolio_vol_target",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable portfolio-level volatility targeting.",
    )

    parser.add_argument(
        "--strategy",
        choices=sorted(strategy_registry().keys()),
        default=None,
        help="Strategy to optimize.",
    )
    parser.add_argument("--metric", default=None, help="Metric used to rank grid search results.")
    parser.add_argument("--hidden-values", dest="hidden_values", type=int, nargs="+", default=None)
    parser.add_argument("--dropout-values", dest="dropout_values", type=float, nargs="+", default=None)
    parser.add_argument("--batch-size-values", dest="batch_size_values", type=int, nargs="+", default=None)
    parser.add_argument("--learning-rate-values", dest="learning_rate_values", type=float, nargs="+", default=None)
    parser.add_argument("--epochs-values", dest="epochs_values", type=int, nargs="+", default=None)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_run_config(args.config) if args.config else RunConfig()
    cfg = merge_cli_overrides(cfg, args)
    cfg = merge_optimization_cli_overrides(cfg, args)

    if cfg.optimization is None:
        parser.error("Optimization config is required. Provide an [optimization] section or CLI overrides.")
    try:
        validate_optimization_config(cfg.optimization)
    except ValueError as exc:
        parser.error(str(exc))

    selected_market = cfg.market or "cac40"
    if cfg.ticker and cfg.market:
        parser.error("Use either --market or --ticker, not both.")
    if cfg.ticker:
        tickers = [cfg.ticker]
    else:
        if selected_market not in MARKET_TICKERS:
            parser.error(f"Unknown market '{selected_market}'. Allowed values: {sorted(MARKET_TICKERS.keys())}")
        tickers = resolve_tickers(selected_market, sector=cfg.sector, sub_sector=cfg.sub_sector)

    if args.print_config:
        payload = cfg.to_dict()
        payload["optimization"] = optimization_config_to_dict(cfg.optimization)
        print(json.dumps(payload, indent=2))

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

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 50)
    display_res = results.copy()
    if "strategy" in display_res.columns:
        display_res = display_res.drop(columns=["strategy"])
    float_cols = display_res.select_dtypes(include=["float64", "float32"]).columns
    display_res[float_cols] = display_res[float_cols].round(3)
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
    print(json.dumps(best, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
