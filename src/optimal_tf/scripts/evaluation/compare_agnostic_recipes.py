from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.scripts.common import build_scenario_highlights, build_scenario_summary, merge_common_overrides, parse_csv_list
from optimal_tf.strategies_agnostic import compute_agnostic_recipe_panel, supported_agnostic_strategies
from trading_core.backtest import compare_strategies
from trading_core.backtest.engine import evaluate_portfolio
from trading_core.risk import estimate_clean_correlation_panel
from trading_core.reporting import render_series_comparison_plot

DEFAULT_RECIPES = ("ARP_AGNOSTIC", "PHI_50", "PHI_100", "ATF_AGNOSTIC")
DEFAULT_OUTPUT_DIR = "output/optimal_tf/evaluation/agnostic_recipes"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare agnostic Eq. 8 recipes on one evaluation run.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    parser.add_argument("--rebalance-frequency", type=str, default=None)
    parser.add_argument("--recipes", type=str, default=",".join(DEFAULT_RECIPES))
    parser.add_argument("--refresh-policy", type=str, default="auto")
    parser.add_argument("--long-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-plot", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_recipes(recipes: list[str]) -> None:
    allowed = supported_agnostic_strategies()
    invalid = [recipe for recipe in recipes if recipe not in allowed]
    if invalid:
        raise ValueError(f"Unknown agnostic recipes {invalid}. Allowed values: {allowed}")


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recipes = parse_csv_list(args.recipes)
    _validate_recipes(recipes)

    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(args.config)
    del allocation, compare, output
    universe, estimation, backtest, evaluation = merge_common_overrides(universe, estimation, backtest, evaluation, args)
    if args.long_only is not None:
        backtest = replace(backtest, long_only=bool(args.long_only))

    prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=args.refresh_policy)

    def _evaluate_recipe(prices_frame, estimation_cfg, backtest_cfg, evaluation_cfg):
        return evaluate_portfolio(
            prices_frame,
            estimation_cfg,
            backtest_cfg,
            evaluation_cfg,
            compute_strategy_panel_fn=lambda p, e, strategy, **kwargs: compute_agnostic_recipe_panel(
                p,
                e,
                strategy,
                long_only=backtest_cfg.long_only,
                target_dates=kwargs.get("target_dates"),
                covariance_cache=kwargs.get("covariance_cache"),
            ),
            estimate_clean_covariance_panel_fn=estimate_clean_correlation_panel,
        )

    result = compare_strategies(
        prices,
        estimation,
        backtest,
        evaluation,
        list(recipes),
        evaluate_portfolio_fn=_evaluate_recipe,
    )

    print(f"universe: {universe.name}")
    print(f"recipes: {', '.join(recipes)}")
    print("scenario_summary:")
    summary = build_scenario_summary(result.summary_table, "strategy")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    highlights = build_scenario_highlights(result.summary_table, "strategy")
    if highlights:
        print("scenario_highlights:")
        for key, value in highlights.items():
            print(f"- {key}: {value}")

    outdir = Path(args.output_dir) if args.output_dir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        result.summary_table.to_csv(outdir / "scenario_summary.csv", index=False)
        result.nav_comparison.to_csv(outdir / "nav_comparison.csv")
        result.drawdown_comparison.to_csv(outdir / "drawdown_comparison.csv")
        if args.output_plot:
            render_series_comparison_plot(
                result.nav_comparison,
                output_path=outdir / "nav_comparison.png",
                title="Agnostic recipe comparison",
                ylabel="NAV",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
