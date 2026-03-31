from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import pandas as pd

from ..allocation import compute_portfolio_strategy_state_at_date, supported_strategies
from ..config import AllocationConfig, BacktestConfig, EstimationConfig, UniverseConfig
from ..config_io import load_config
from ..data import load_prices_for_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute optimal_tf portfolio weights on a given allocation date.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("configs/optimal_tf.example.toml")),
        help="Path to a TOML config file.",
    )
    parser.add_argument("--universe", type=str, default=None, help="Universe name from market_tickers_data.")
    parser.add_argument("--start", type=str, default=None, help="Start date for price history.")
    parser.add_argument("--date", type=str, default=None, help="Allocation date. Defaults to today.")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=supported_strategies(),
        help="Portfolio recipe to use.",
    )
    parser.add_argument(
        "--long-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Project weights to long-only after the portfolio recipe.",
    )
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to save the weights as CSV.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save the weights as JSON.")
    return parser


def _merge_overrides(
    universe: UniverseConfig,
    estimation: EstimationConfig,
    backtest: BacktestConfig,
    allocation: AllocationConfig,
    args: argparse.Namespace,
) -> tuple[UniverseConfig, EstimationConfig, BacktestConfig, AllocationConfig]:
    if args.universe is not None:
        universe = UniverseConfig(name=args.universe, start=universe.start)
    if args.start is not None:
        universe = UniverseConfig(name=universe.name, start=args.start)
    if args.long_only is not None:
        backtest = BacktestConfig(
            sigma_target_annual=backtest.sigma_target_annual,
            portfolio_vol_target=backtest.portfolio_vol_target,
            portfolio_vol_span=backtest.portfolio_vol_span,
            cost_bps=backtest.cost_bps,
            long_only=args.long_only,
        )
    if args.strategy is not None or args.date is not None:
        allocation = AllocationConfig(
            strategy=args.strategy if args.strategy is not None else allocation.strategy,
            date=args.date if args.date is not None else allocation.date,
        )
    return universe, estimation, backtest, allocation


def _format_weights(weights: pd.Series) -> str:
    display = weights[weights != 0.0].sort_values(ascending=False)
    if display.empty:
        display = weights.sort_values(ascending=False)
    return display.to_string(float_format=lambda x: f"{x: .6f}")


def _write_outputs(
    weights: pd.Series,
    base_weights: pd.Series,
    signal_scale: float,
    allocation_date: pd.Timestamp,
    strategy: str,
    universe: str,
    csv_path: str | None,
    json_path: str | None,
) -> None:
    export = weights.rename("weight").reset_index().rename(columns={"index": "ticker"})
    export.insert(0, "date", allocation_date.strftime("%Y-%m-%d"))
    export.insert(1, "strategy", strategy)
    export.insert(2, "universe", universe)

    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export.to_csv(path, index=False)

    if json_path is not None:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "date": allocation_date.strftime("%Y-%m-%d"),
            "strategy": strategy,
            "universe": universe,
            "signal_scale": float(signal_scale),
            "base_weights": {str(k): float(v) for k, v in base_weights.items()},
            "weights": {str(k): float(v) for k, v in weights.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    started_at = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)

    universe, estimation, backtest, allocation, _ = load_config(args.config)
    universe, estimation, backtest, allocation = _merge_overrides(universe, estimation, backtest, allocation, args)

    prices = load_prices_for_universe(universe.name, start=universe.start)
    allocation_date, state = compute_portfolio_strategy_state_at_date(
        prices,
        estimation,
        allocation.strategy,
        as_of_date=allocation.date,
        long_only=backtest.long_only,
    )
    weights = state.effective_weights

    print(f"strategy: {allocation.strategy}")
    print(f"universe: {universe.name}")
    print(f"requested_date: {pd.Timestamp(allocation.date).date() if allocation.date else pd.Timestamp.today().date()}")
    print(f"allocation_date: {allocation_date.date()}")
    print(f"signal_scale: {state.signal_scale: .6f}")
    print(f"num_assets: {(weights != 0.0).sum()}")
    print(f"execution_time_seconds: {perf_counter() - started_at: .3f}")
    print(_format_weights(weights))
    _write_outputs(
        weights,
        state.base_weights,
        state.signal_scale,
        allocation_date,
        allocation.strategy,
        universe.name,
        args.output_csv,
        args.output_json,
    )
    return 0
