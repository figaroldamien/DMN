from __future__ import annotations

import argparse
from typing import Sequence

from ..backtest import backtest_all
from ..strategies import strategy_names
from .common import (
    RESULT_COLUMN_RENAMES,
    build_parser_with_argsets,
    format_results_table,
    load_effective_run_config,
    print_config_payload,
    resolve_config_tickers,
    tee_output,
)
from ..data import load_prices_yf


def build_parser() -> argparse.ArgumentParser:
    parser = build_parser_with_argsets(
        "Backtest TSMOM/DMN strategies on a predefined market universe.",
        "config",
        "universe",
        "backtest",
        "model",
    )

    parser.add_argument("--run-ml", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable ML baselines.")
    parser.add_argument("--run-dmn", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable DMN LSTM strategy.")
    parser.epilog = f"Available optimization-capable sequence strategies: {', '.join(strategy_names(supports_optimization=True))}"
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    with tee_output("backtest"):
        cfg = load_effective_run_config(args)
        tickers = resolve_config_tickers(cfg, parser)

        if args.print_config:
            print_config_payload(cfg.to_dict())
        try:
            prices = load_prices_yf(tickers, start=cfg.start)
        except ImportError:
            print("Install yfinance to run the example: pip install yfinance")
            return 0

        res = backtest_all(
            prices,
            cfg.backtest,
            run_ml=cfg.run_ml,
            run_dmn=cfg.run_dmn,
            model=cfg.model,
        )
        display_res = format_results_table(res, width=140, rename_columns=RESULT_COLUMN_RENAMES)
        print(display_res.to_string(index=False, col_space=5))
        return 0
