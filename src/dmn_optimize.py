"""Minimal Python entrypoint for hyperparameter grid search.

Examples:
  python src/dmn_optimize.py --market cac40 --strategy DMN_LSTM_Sharpe_TurnPen
  python src/dmn_optimize.py --ticker AAPL --strategy VLSTM_Sharpe

All other optimization and backtest parameters are loaded from the TOML config.
"""

from __future__ import annotations

import argparse

from dmn.cli.optimize_cli import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization with strategy and market/ticker only."
    )
    parser.add_argument(
        "--config",
        default="src/dmn/cli/config.example.toml",
        help="Path to the TOML/JSON config file containing the optimization grid.",
    )
    parser.add_argument("--market", default=None, help="Ticker universe to load.")
    parser.add_argument("--ticker", default=None, help="Single ticker to load directly (exclusive with --market).")
    parser.add_argument("--sector", default=None, help="Optional sector filter for component-based indices.")
    parser.add_argument("--sub-sector", dest="sub_sector", default=None, help="Optional sub-sector filter.")
    parser.add_argument("--strategy", required=True, help="Strategy to optimize.")
    parser.add_argument(
        "--no-print-config",
        dest="print_config",
        action="store_false",
        default=True,
        help="Disable effective configuration display.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    #args = parser.parse_args(["--strategy", "VLSTM_Sharpe_TurnPen", 
    #                          "--market", "nasdaq100", 
    #                          "--sector", "Technology", 
    #                          "--sub-sector", "Software"])
    
    args = parser.parse_args(["--strategy", "DMN_LSTM_Sharpe_TurnPen", 
                              "--ticker", "AAPL", 
                              "--sector", "Technology", 
                              "--sub-sector", "Software"])

    argv = ["--config", args.config, "--strategy", args.strategy]
    if args.market and args.ticker:
        parser.error("Use either --market or --ticker, not both.")
    if args.market:
        argv.extend(["--market", args.market])
    if args.ticker:
        argv.extend(["--ticker", args.ticker])
    market_overridden = args.market is not None
    if args.sector:
        argv.extend(["--sector", args.sector])
    elif market_overridden:
        argv.extend(["--sector", ""])
    if args.sub_sector:
        argv.extend(["--sub-sector", args.sub_sector])
    elif market_overridden:
        argv.extend(["--sub-sector", ""])
    if not args.print_config:
        argv.append("--no-print-config")

    raise SystemExit(run(argv))
