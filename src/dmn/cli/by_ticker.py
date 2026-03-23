from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from ..backtest import backtest_strategy
from ..strategies import get_strategy_spec
from .common import (
    RESULT_COLUMN_RENAMES,
    add_strategy_choice_arg,
    build_parser_with_argsets,
    format_results_table,
    infer_strategy_argsets,
    load_effective_run_config,
    print_config_payload,
    resolve_config_tickers,
    strategy_choice_names,
    tee_output,
)
from ..data import load_prices_yf


def build_parser() -> argparse.ArgumentParser:
    strategy_choices = strategy_choice_names()
    parser = build_parser_with_argsets(
        "Evaluate one strategy independently for each ticker in a selected universe.",
        "config",
        "universe",
        "backtest",
        *infer_strategy_argsets(strategy_choices, allowed_argsets={"model"}),
    )

    parser.add_argument("--run-ml", action=argparse.BooleanOptionalAction, default=None, help="Kept for argument compatibility with dmn_test; ignored here.")
    parser.add_argument("--run-dmn", action=argparse.BooleanOptionalAction, default=None, help="Kept for argument compatibility with dmn_test; ignored here.")
    add_strategy_choice_arg(
        parser,
        required=True,
        help_text="Strategy to apply independently on each ticker.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    with tee_output("by_ticker"):
        cfg = load_effective_run_config(args)
        tickers = resolve_config_tickers(cfg, parser)

        if args.print_config:
            payload = cfg.to_dict()
            payload["strategy"] = args.strategy
            print_config_payload(payload)

        try:
            prices = load_prices_yf(tickers, start=cfg.start)
        except ImportError:
            print("Install yfinance to run the example: pip install yfinance")
            return 0

        strategy_spec = get_strategy_spec(args.strategy)
        strategy_fn = strategy_spec.fn
        default_kwargs = dict(strategy_spec.default_kwargs)
        results = []
        for ticker in tickers:
            px = prices[[ticker]].dropna(how="all")
            if px.empty:
                continue

            kwargs = dict(default_kwargs)
            if "model" in strategy_spec.cli_argsets:
                row = backtest_strategy(args.strategy, strategy_fn, px, cfg.backtest, px, cfg.backtest, **kwargs)
            elif args.strategy.startswith("ML_"):
                row = backtest_strategy(args.strategy, strategy_fn, px, cfg.backtest, px, cfg.backtest, **kwargs)
            else:
                row = backtest_strategy(args.strategy, strategy_fn, px, cfg.backtest, px, **kwargs)

            row = row.copy()
            row.insert(0, "ticker", ticker)
            results.append(row)

        if not results:
            print("No results to display (no ticker with usable data).")
            return 0

        res = pd.concat(results, ignore_index=True).sort_values("sharpe", ascending=False).reset_index(drop=True)

        display_res = format_results_table(res, rename_columns=RESULT_COLUMN_RENAMES)
        print(display_res.to_string(index=False, col_space=5))
        return 0


if __name__ == "__main__":
    raise SystemExit(run())
