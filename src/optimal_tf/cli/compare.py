from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Sequence

from ..comparison import compare_strategies
from ..config import AllocationConfig, BacktestConfig, EstimationConfig, EvaluationConfig, UniverseConfig
from ..config_io import load_config
from ..data import load_prices_for_universe
from ..rebalance import supported_rebalance_frequencies
from ..reporting import render_series_comparison_plot
from .evaluate import _write_outputs as write_evaluation_outputs
from ..allocation import supported_strategies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare several optimal_tf strategies on the same evaluation run.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")), help="Path to a TOML config file.")
    parser.add_argument("--universe", type=str, default=None, help="Universe name from market_tickers_data.")
    parser.add_argument("--start", type=str, default=None, help="Start date for price history.")
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategies to compare. Defaults to allocation/evaluation strategy if omitted.",
    )
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
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for comparison outputs.")
    parser.add_argument(
        "--clean-output-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean the output directory before writing comparison results.",
    )
    parser.add_argument(
        "--output-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate comparison PNG charts.",
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
    frequency = args.rebalance_frequency if args.rebalance_frequency is not None else evaluation.rebalance_frequency
    eval_start = args.evaluation_start if args.evaluation_start is not None else evaluation.evaluation_start
    eval_end = args.evaluation_end if args.evaluation_end is not None else evaluation.evaluation_end
    evaluation = EvaluationConfig(
        strategy=evaluation.strategy,
        rebalance_frequency=frequency,
        evaluation_start=eval_start,
        evaluation_end=eval_end,
    )
    return universe, estimation, backtest, allocation, evaluation


def _resolve_strategies(raw: str | None, allocation: AllocationConfig, evaluation: EvaluationConfig) -> list[str]:
    if raw is None:
        return [evaluation.strategy or allocation.strategy]
    strategies = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in strategies if item not in supported_strategies()]
    if invalid:
        raise ValueError(f"Unknown strategies {invalid}. Allowed values: {supported_strategies()}")
    if not strategies:
        raise ValueError("At least one strategy must be provided.")
    return strategies


def _write_strategy_outputs(outdir: Path, strategy_results: dict) -> None:
    strategies_dir = outdir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    for strategy, result in strategy_results.items():
        write_evaluation_outputs(result, str(strategies_dir / strategy))


def _prepare_output_dir(outdir: Path, *, clean: bool) -> None:
    if clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)


def _write_comparison_outputs(outdir: Path, comparison) -> None:
    comp_dir = outdir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    comparison.summary_table.to_csv(comp_dir / "summary_table.csv", index=False)
    comparison.nav_comparison.to_csv(comp_dir / "nav_comparison.csv")
    comparison.drawdown_comparison.to_csv(comp_dir / "drawdown_comparison.csv")


def _write_manifest(
    outdir: Path,
    universe: UniverseConfig,
    backtest: BacktestConfig,
    evaluation: EvaluationConfig,
    strategies: list[str],
) -> None:
    manifest = {
        "universe": universe.name,
        "start": universe.start,
        "rebalance_frequency": evaluation.rebalance_frequency,
        "evaluation_start": evaluation.evaluation_start,
        "evaluation_end": evaluation.evaluation_end,
        "long_only": backtest.long_only,
        "strategies": strategies,
        "available_views": [
            "summary_table",
            "nav_comparison",
            "drawdown_comparison",
            "strategy_detail",
        ],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_inputs(
    outdir: Path,
    universe: UniverseConfig,
    estimation: EstimationConfig,
    backtest: BacktestConfig,
    allocation: AllocationConfig,
    evaluation: EvaluationConfig,
) -> None:
    payload = {
        "universe": asdict(universe),
        "estimation": asdict(estimation),
        "backtest": asdict(backtest),
        "allocation": asdict(allocation),
        "evaluation": asdict(evaluation),
    }
    (outdir / "inputs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_plots(outdir: Path, comparison) -> None:
    plots_dir = outdir / "comparison" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    render_series_comparison_plot(
        comparison.nav_comparison,
        plots_dir / "nav_comparison.png",
        title="Strategy NAV Comparison",
        ylabel="Cumulative value",
    )
    render_series_comparison_plot(
        comparison.drawdown_comparison,
        plots_dir / "drawdown_comparison.png",
        title="Strategy Drawdown Comparison",
        ylabel="Drawdown",
    )


def run(argv: Sequence[str] | None = None) -> int:
    started_at = perf_counter()
    parser = build_parser()
    args = parser.parse_args(argv)

    universe, estimation, backtest, allocation, evaluation = load_config(args.config)
    universe, estimation, backtest, allocation, evaluation = _merge_overrides(
        universe, estimation, backtest, allocation, evaluation, args
    )
    strategies = _resolve_strategies(args.strategies, allocation, evaluation)
    prices = load_prices_for_universe(universe.name, start=universe.start)
    comparison = compare_strategies(prices, estimation, backtest, evaluation, strategies)

    outdir = Path(args.output_dir)
    _prepare_output_dir(outdir, clean=args.clean_output_dir)
    _write_strategy_outputs(outdir, comparison.strategy_results)
    _write_comparison_outputs(outdir, comparison)
    _write_manifest(outdir, universe, backtest, evaluation, strategies)
    _write_inputs(outdir, universe, estimation, backtest, allocation, evaluation)
    if args.output_plot:
        _render_plots(outdir, comparison)

    print(f"universe: {universe.name}")
    print(f"rebalance_frequency: {evaluation.rebalance_frequency}")
    print(f"strategies: {', '.join(strategies)}")
    print(f"evaluation_start: {evaluation.evaluation_start or prices.index.min().date()}")
    print(f"evaluation_end: {evaluation.evaluation_end or prices.index.max().date()}")
    print(f"execution_time_seconds: {perf_counter() - started_at: .3f}")
    if not comparison.summary_table.empty:
        print("summary:")
        print(comparison.summary_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return 0
