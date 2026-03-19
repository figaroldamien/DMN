from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from market_tickers import MARKET_TICKERS

from ..config import RunConfig
from ..config_io import load_run_config, merge_cli_overrides, merge_optimization_cli_overrides
from ..strategies import strategy_names, strategy_specs
from ..universe import resolve_tickers


RESULT_COLUMN_RENAMES = {
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

ARGSET_DEFAULTS: dict[str, dict[str, Any]] = {
    "config": {},
    "universe": {
        "mutually_exclusive": False,
        "start_default": None,
    },
    "backtest": {
        "sigma_target_annual_default": None,
        "vol_span_default": None,
        "cost_bps_default": None,
        "min_obs_default": None,
        "portfolio_vol_target_default": None,
    },
    "model": {},
    "optimization": {},
}


def add_config_args(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    parser.add_argument("--config", default=None, help="Optional path to a TOML/JSON configuration file.")
    parser.add_argument(
        "--no-print-config",
        dest="print_config",
        action="store_false",
        default=True,
        help="Disable effective configuration display.",
    )


def add_universe_args(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    *,
    mutually_exclusive: bool = False,
    start_default: str | None = None,
) -> None:
    target: argparse.ArgumentParser | argparse._ArgumentGroup = parser
    if mutually_exclusive:
        target = parser.add_mutually_exclusive_group()
    target.add_argument("--market", default=None, choices=sorted(MARKET_TICKERS.keys()), help="Ticker universe to load.")
    target.add_argument("--ticker", default=None, help="Single ticker to load directly (exclusive with --market).")
    parser.add_argument("--start", default=start_default, help="Start date for yfinance download (YYYY-MM-DD).")
    parser.add_argument("--sector", default=None, help="Optional sector filter (for component-based indices).")
    parser.add_argument("--sub-sector", dest="sub_sector", default=None, help="Optional sub-sector filter (for component-based indices).")


def add_backtest_args(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    *,
    sigma_target_annual_default: float | None = None,
    vol_span_default: int | None = None,
    cost_bps_default: float | None = None,
    min_obs_default: int | None = None,
    portfolio_vol_target_default: bool | None = None,
) -> None:
    parser.add_argument("--sigma-target-annual", type=float, default=sigma_target_annual_default, help="Target annualized volatility.")
    parser.add_argument("--vol-span", type=int, default=vol_span_default, help="EWMA span for daily volatility.")
    parser.add_argument("--cost-bps", type=float, default=cost_bps_default, help="Transaction cost in bps per turnover unit.")
    parser.add_argument("--min-obs", type=int, default=min_obs_default, help="Minimum warmup observations.")
    parser.add_argument(
        "--portfolio-vol-target",
        dest="portfolio_vol_target",
        action=argparse.BooleanOptionalAction,
        default=portfolio_vol_target_default,
        help="Enable/disable portfolio-level volatility targeting.",
    )


def add_model_args(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    parser.add_argument(
        "--hidden",
        "--dmn-hidden",
        "--model-hidden",
        dest="model_hidden",
        type=int,
        default=None,
        help="Hidden size for DMN sequence models.",
    )
    parser.add_argument(
        "--dropout",
        "--dmn-dropout",
        "--model-dropout",
        dest="model_dropout",
        type=float,
        default=None,
        help="Dropout for DMN sequence models.",
    )
    parser.add_argument(
        "--use-ticker-embedding",
        "--dmn-use-ticker-embedding",
        "--model-use-ticker-embedding",
        dest="model_use_ticker_embedding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable ticker embedding for DMN sequence models.",
    )


def add_optimization_args(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    parser.add_argument("--metric", default=None, help="Metric used to rank grid search results.")
    parser.add_argument("--hidden-values", dest="hidden_values", type=int, nargs="+", default=None)
    parser.add_argument("--dropout-values", dest="dropout_values", type=float, nargs="+", default=None)
    parser.add_argument("--batch-size-values", dest="batch_size_values", type=int, nargs="+", default=None)
    parser.add_argument("--learning-rate-values", dest="learning_rate_values", type=float, nargs="+", default=None)
    parser.add_argument("--epochs-values", dest="epochs_values", type=int, nargs="+", default=None)


ARGSET_BUILDERS = {
    "config": add_config_args,
    "universe": add_universe_args,
    "backtest": add_backtest_args,
    "model": add_model_args,
    "optimization": add_optimization_args,
}


def apply_argsets(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    *argsets: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    overrides = overrides or {}
    for argset in argsets:
        if argset not in ARGSET_BUILDERS:
            raise ValueError(f"Unknown CLI argset '{argset}'. Allowed values: {sorted(ARGSET_BUILDERS)}")
        kwargs = dict(ARGSET_DEFAULTS.get(argset, {}))
        kwargs.update(overrides.get(argset, {}))
        ARGSET_BUILDERS[argset](parser, **kwargs)


def build_parser_with_argsets(
    description: str,
    *argsets: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    apply_argsets(parser, *argsets, overrides=overrides)
    return parser


def infer_strategy_argsets(
    strategy_choices: list[str],
    *,
    allowed_argsets: set[str] | None = None,
) -> list[str]:
    inferred: list[str] = []
    seen: set[str] = set()
    specs = strategy_specs()
    for strategy_name in strategy_choices:
        spec = specs[strategy_name]
        for argset in spec.cli_argsets:
            if allowed_argsets is not None and argset not in allowed_argsets:
                continue
            if argset not in seen:
                inferred.append(argset)
                seen.add(argset)
    return inferred


def strategy_choice_names(*, supports_optimization: bool | None = None) -> list[str]:
    return strategy_names(supports_optimization=supports_optimization)


def add_strategy_choice_arg(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    *,
    required: bool = False,
    supports_optimization: bool | None = None,
    help_text: str,
) -> list[str]:
    choices = strategy_choice_names(supports_optimization=supports_optimization)
    parser.add_argument(
        "--strategy",
        required=required,
        choices=choices,
        default=None,
        help=help_text,
    )
    return choices


def load_effective_run_config(
    args: argparse.Namespace,
    *,
    include_optimization: bool = False,
) -> RunConfig:
    cfg = load_run_config(args.config) if getattr(args, "config", None) else RunConfig()
    cfg = merge_cli_overrides(cfg, args)
    if include_optimization:
        cfg = merge_optimization_cli_overrides(cfg, args)
    return cfg


def resolve_config_tickers(
    cfg: RunConfig,
    parser: argparse.ArgumentParser,
) -> list[str]:
    selected_market = cfg.market or "cac40"
    if cfg.ticker and cfg.market:
        parser.error("Use either --market or --ticker, not both.")
    if cfg.ticker:
        return [cfg.ticker]
    if selected_market not in MARKET_TICKERS:
        parser.error(f"Unknown market '{selected_market}'. Allowed values: {sorted(MARKET_TICKERS.keys())}")
    return resolve_tickers(selected_market, sector=cfg.sector, sub_sector=cfg.sub_sector)


def print_config_payload(payload: dict[str, Any], *, default: Any | None = None) -> None:
    print(json.dumps(payload, indent=2, default=default))


def format_results_table(
    df: pd.DataFrame,
    *,
    width: int = 160,
    max_columns: int = 50,
    round_digits: int = 3,
    rename_columns: dict[str, str] | None = None,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    pd.set_option("display.width", width)
    pd.set_option("display.max_columns", max_columns)

    display_res = df.copy()
    if drop_columns:
        existing = [column for column in drop_columns if column in display_res.columns]
        if existing:
            display_res = display_res.drop(columns=existing)

    float_cols = display_res.select_dtypes(include=["float64", "float32"]).columns
    display_res[float_cols] = display_res[float_cols].round(round_digits)

    if rename_columns:
        display_res = display_res.rename(columns=rename_columns)

    return display_res
