from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ..allocation import supported_strategies
from ..config import AllocationConfig, BacktestConfig, EstimationConfig, EvaluationConfig, UniverseConfig
from ..config_io import load_config
from ..data import load_prices_for_universe
from ..evaluation import evaluate_portfolio
from ..rebalance import supported_rebalance_frequencies
from ..reporting import equal_weight_buy_and_hold_benchmark, equal_weight_rebalanced_benchmark, render_evaluation_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a periodic evaluation of optimal_tf portfolios.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")), help="Path to a TOML config file.")
    parser.add_argument("--universe", type=str, default=None, help="Universe name from market_tickers_data.")
    parser.add_argument("--start", type=str, default=None, help="Start date for price history.")
    parser.add_argument("--strategy", type=str, default=None, choices=supported_strategies(), help="Portfolio recipe to use.")
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default=None,
        choices=supported_rebalance_frequencies(),
        help="Portfolio rebalance schedule.",
    )
    parser.add_argument("--evaluation-start", type=str, default=None, help="Start date for the evaluation window.")
    parser.add_argument("--evaluation-end", type=str, default=None, help="End date for the evaluation window.")
    parser.add_argument("--long-only", action=argparse.BooleanOptionalAction, default=None, help="Project weights to long-only.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for CSV/JSON exports.")
    parser.add_argument(
        "--output-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate a PNG performance chart when --output-dir is provided.",
    )
    return parser


def _merge_overrides(
    universe: UniverseConfig,
    estimation: EstimationConfig,
    backtest: BacktestConfig,
    allocation: AllocationConfig,
    evaluation: EvaluationConfig,
    args: argparse.Namespace,
) -> tuple[UniverseConfig, EstimationConfig, BacktestConfig, AllocationConfig, EvaluationConfig]:
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
    strategy = args.strategy if args.strategy is not None else evaluation.strategy
    frequency = args.rebalance_frequency if args.rebalance_frequency is not None else evaluation.rebalance_frequency
    eval_start = args.evaluation_start if args.evaluation_start is not None else evaluation.evaluation_start
    eval_end = args.evaluation_end if args.evaluation_end is not None else evaluation.evaluation_end
    allocation = AllocationConfig(strategy=strategy, date=allocation.date)
    evaluation = EvaluationConfig(
        strategy=strategy,
        rebalance_frequency=frequency,
        evaluation_start=eval_start,
        evaluation_end=eval_end,
    )
    return universe, estimation, backtest, allocation, evaluation


def _write_outputs(result, output_dir: str | None) -> None:
    if output_dir is None:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.weights_by_rebalance.to_csv(out / "weights_by_rebalance.csv")
    if not result.base_weights_by_rebalance.empty:
        result.base_weights_by_rebalance.to_csv(out / "base_weights_by_rebalance.csv")
    if not result.effective_weights_by_rebalance.empty:
        result.effective_weights_by_rebalance.to_csv(out / "effective_weights_by_rebalance.csv")
    if len(result.signal_scale_by_rebalance) > 0:
        result.signal_scale_by_rebalance.rename("signal_scale").to_csv(out / "signal_scale.csv", header=True)
    if len(result.portfolio_vol_scale) > 0:
        result.portfolio_vol_scale.rename("portfolio_vol_scale").to_csv(out / "portfolio_vol_scale.csv", header=True)
    result.daily_returns_gross.rename("gross_return").to_csv(out / "daily_returns_gross.csv", header=True)
    result.daily_returns_net.rename("net_return").to_csv(out / "daily_returns_net.csv", header=True)
    result.turnover_by_rebalance.rename("turnover").to_csv(out / "turnover.csv", header=True)
    result.costs_by_rebalance.rename("cost").to_csv(out / "costs.csv", header=True)
    (out / "summary.json").write_text(json.dumps(asdict(result.summary), indent=2), encoding="utf-8")


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    universe, estimation, backtest, allocation, evaluation = load_config(args.config)
    universe, estimation, backtest, allocation, evaluation = _merge_overrides(
        universe, estimation, backtest, allocation, evaluation, args
    )

    prices = load_prices_for_universe(universe.name, start=universe.start)
    result = evaluate_portfolio(prices, estimation, backtest, evaluation)

    print(f"strategy: {evaluation.strategy}")
    print(f"universe: {universe.name}")
    print(f"rebalance_frequency: {evaluation.rebalance_frequency}")
    print(f"evaluation_start: {evaluation.evaluation_start or prices.index.min().date()}")
    print(f"evaluation_end: {evaluation.evaluation_end or prices.index.max().date()}")
    for key, value in asdict(result.summary).items():
        print(f"{key}: {value}")
    _write_outputs(result, args.output_dir)
    if args.output_dir is not None and args.output_plot:
        benchmark_returns = equal_weight_rebalanced_benchmark(
            prices,
            max_abs_return=estimation.max_abs_return,
        ).loc[result.daily_returns_net.index]
        buy_hold_returns = equal_weight_buy_and_hold_benchmark(
            prices,
            max_abs_return=estimation.max_abs_return,
        ).loc[result.daily_returns_net.index]
        plot_path = render_evaluation_plot(
            result.daily_returns_net,
            benchmark_returns,
            buy_hold_returns,
            Path(args.output_dir) / "performance.png",
            title=f"{evaluation.strategy} vs universe benchmarks ({universe.name})",
        )
        print(f"plot: {plot_path}")
    return 0
