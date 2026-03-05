from __future__ import annotations

import argparse
import json
from typing import Sequence

import pandas as pd

from market_tickers import MARKET_TICKERS

from ..config import RunConfig
from ..config_io import load_run_config, merge_cli_overrides
from ..data import load_prices_yf
from ..runner import backtest_all
from ..universe import resolve_tickers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest TSMOM/DMN strategies on a predefined market universe.")
    parser.add_argument("--config", default=None, help="Optional path to a TOML/JSON configuration file.")
    parser.add_argument("--no-print-config", dest="print_config", action="store_false", default=True, help="Disable effective configuration display.")
    parser.add_argument("--market", default=None, choices=sorted(MARKET_TICKERS.keys()), help="Ticker universe to load.")
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

    parser.add_argument("--run-ml", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable ML baselines.")
    parser.add_argument("--run-dmn", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable DMN LSTM strategy.")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_run_config(args.config) if args.config else RunConfig()
    cfg = merge_cli_overrides(cfg, args)
    if cfg.market not in MARKET_TICKERS:
        parser.error(f"Unknown market '{cfg.market}'. Allowed values: {sorted(MARKET_TICKERS.keys())}")

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))

    tickers = resolve_tickers(cfg.market, sector=cfg.sector, sub_sector=cfg.sub_sector)
    try:
        prices = load_prices_yf(tickers, start=cfg.start)
    except ImportError:
        print("Install yfinance to run the example: pip install yfinance")
        return 0

    res = backtest_all(prices, cfg.backtest, run_ml=cfg.run_ml, run_dmn=cfg.run_dmn)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    print(res)
    return 0
