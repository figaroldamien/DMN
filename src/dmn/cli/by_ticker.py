from __future__ import annotations

import argparse
import json
from typing import Callable, Sequence

import pandas as pd

from market_tickers import MARKET_TICKERS

from ..backtest import backtest_strategy
from ..config import RunConfig
from ..config_io import load_run_config, merge_cli_overrides
from ..data import load_prices_yf
from ..strategies import (
    dmn_lstm_positions,
    ml_supervised_positions,
    strategy_baz_macd,
    strategy_long_only,
    strategy_sgn_12m,
    vlstm_positions,
    xlstm_positions,
)
from ..universe import resolve_tickers


def _strategy_registry() -> dict[str, tuple[Callable[..., pd.DataFrame], dict]]:
    return {
        "LongOnly": (strategy_long_only, {}),
        "Sgn12M": (strategy_sgn_12m, {}),
        "MACD_Baz": (strategy_baz_macd, {}),
        "ML_LassoReg": (ml_supervised_positions, {"model_type": "lasso_reg"}),
        "ML_MLPReg": (ml_supervised_positions, {"model_type": "mlp_reg"}),
        "ML_LassoClf": (ml_supervised_positions, {"model_type": "lasso_clf"}),
        "DMN_LSTM_Sharpe": (dmn_lstm_positions, {"turnover_lambda": 0.0}),
        "DMN_LSTM_Sharpe_TurnPen": (dmn_lstm_positions, {"turnover_lambda": 1e-2}),
        "VLSTM_Sharpe": (vlstm_positions, {"turnover_lambda": 0.0}),
        "VLSTM_Sharpe_TurnPen": (vlstm_positions, {"turnover_lambda": 1e-2}),
        "xLSTM_Sharpe": (xlstm_positions, {"turnover_lambda": 0.0}),
        "xLSTM_Sharpe_TurnPen": (xlstm_positions, {"turnover_lambda": 1e-2}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one strategy independently for each ticker in a selected universe."
    )
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

    parser.add_argument("--run-ml", action=argparse.BooleanOptionalAction, default=None, help="Kept for argument compatibility with dmn_test; ignored here.")
    parser.add_argument("--run-dmn", action=argparse.BooleanOptionalAction, default=None, help="Kept for argument compatibility with dmn_test; ignored here.")

    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(_strategy_registry().keys()),
        help="Strategy to apply independently on each ticker.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_run_config(args.config) if args.config else RunConfig()
    cfg = merge_cli_overrides(cfg, args)

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
        payload["strategy"] = args.strategy
        print(json.dumps(payload, indent=2))

    try:
        prices = load_prices_yf(tickers, start=cfg.start)
    except ImportError:
        print("Install yfinance to run the example: pip install yfinance")
        return 0

    strategy_fn, default_kwargs = _strategy_registry()[args.strategy]
    results = []
    for ticker in tickers:
        px = prices[[ticker]].dropna(how="all")
        if px.empty:
            continue

        kwargs = dict(default_kwargs)
        if strategy_fn in (ml_supervised_positions, dmn_lstm_positions, vlstm_positions, xlstm_positions):
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

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)
    display_res = res.copy()
    float_cols = display_res.select_dtypes(include=["float64", "float32"]).columns
    display_res[float_cols] = display_res[float_cols].round(3)
    display_res = display_res.rename(
        columns={
            "ticker": "tkr",
            "strategy": "strat",
            "ann_return": "ret",
            "ann_vol": "vol",
            "sharpe": "shp",
            "sortino": "sor",
            "calmar": "cal",
            "mdd": "mdd",
            "pct_pos": "pos",
            "avgP_over_avgL": "p_l",
            "avg_turnover": "turn",
            "elapsed_s": "sec",
        }
    )
    print(display_res.to_string(index=False, col_space=5))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
