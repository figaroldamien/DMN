from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.scripts.common import build_scenario_highlights, merge_common_overrides
from optimal_tf.strategies_agnostic import compute_agnostic_panel
from trading_core.backtest import compare_strategies
from trading_core.backtest.engine import evaluate_portfolio
from trading_core.reporting import render_series_comparison_plot
from trading_core.risk import estimate_clean_correlation_panel

DEFAULT_WINDOWS = "21,63,126,252"
DEFAULT_PHI = "0.0,0.5,1.0"
DEFAULT_OUTPUT_DIR = "output/optimal_tf/evaluation/atf_trend_sensitivity"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sensitivity study for ATF_AGNOSTIC across trend EMA windows and phi values.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--covariance-window", type=int, default=None)
    parser.add_argument("--covariance-min-periods", type=int, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    parser.add_argument("--rebalance-frequency", type=str, default=None)
    parser.add_argument("--trend-windows", type=str, default=DEFAULT_WINDOWS, help="Comma-separated EMA spans in trading days.")
    parser.add_argument("--phi-values", type=str, default=DEFAULT_PHI, help="Comma-separated phi values.")
    parser.add_argument("--refresh-policy", type=str, default="auto")
    parser.add_argument("--long-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-plot", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 1 for value in values):
        raise ValueError(f"trend windows must be integers > 1 (got {raw!r}).")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"phi values must lie in [0, 1] (got {raw!r}).")
    return values


def _label(window: int, phi: float) -> str:
    return f"ATF_EMA_{window}_PHI_{phi:.1f}"


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    windows = _parse_int_list(args.trend_windows)
    phi_values = _parse_float_list(args.phi_values)
    labels = [_label(window, phi) for window in windows for phi in phi_values]
    params = {label: (window, phi) for window in windows for phi in phi_values for label in [_label(window, phi)]}

    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(args.config)
    del allocation, compare, output
    universe, estimation, backtest, evaluation = merge_common_overrides(universe, estimation, backtest, evaluation, args)
    if args.long_only is not None:
        backtest = replace(backtest, long_only=bool(args.long_only))

    prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=args.refresh_policy)

    def _evaluate_atf(prices_frame, estimation_cfg, backtest_cfg, evaluation_cfg):
        label = str(evaluation_cfg.strategy)
        window, phi = params[label]
        local_estimation = replace(estimation_cfg, trend_alpha=None, trend_span=window)
        return evaluate_portfolio(
            prices_frame,
            local_estimation,
            backtest_cfg,
            evaluation_cfg,
            compute_strategy_panel_fn=lambda p, e, strategy, **kwargs: compute_agnostic_panel(
                p,
                e,
                signal_model="trend_ema",
                q_model="phi_shrink_correlation",
                phi=phi,
                normalization="gross",
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
        labels,
        evaluate_portfolio_fn=_evaluate_atf,
    )

    summary = result.summary_table.copy()
    summary["trend_window"] = summary["strategy"].map(lambda value: params[str(value)][0])
    summary["phi"] = summary["strategy"].map(lambda value: params[str(value)][1])
    ordered_cols = ["strategy", "trend_window", "phi", *[col for col in summary.columns if col not in {"strategy", "trend_window", "phi"}]]
    summary = summary.loc[:, ordered_cols].sort_values(["trend_window", "phi"]).reset_index(drop=True)

    print(f"universe: {universe.name}")
    print(f"trend_windows: {', '.join(str(value) for value in windows)}")
    print(f"phi_values: {', '.join(f'{value:.1f}' for value in phi_values)}")
    print("scenario_summary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    highlights = build_scenario_highlights(summary, "strategy")
    if highlights:
        print("scenario_highlights:")
        for key, value in highlights.items():
            print(f"- {key}: {value}")

    outdir = Path(args.output_dir) if args.output_dir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(outdir / "scenario_summary.csv", index=False)
        sharpe_pivot = summary.pivot(index="trend_window", columns="phi", values="sharpe")
        total_return_pivot = summary.pivot(index="trend_window", columns="phi", values="total_return")
        sharpe_pivot.to_csv(outdir / "sharpe_by_window_phi.csv")
        total_return_pivot.to_csv(outdir / "total_return_by_window_phi.csv")
        result.nav_comparison.to_csv(outdir / "nav_comparison.csv")
        result.drawdown_comparison.to_csv(outdir / "drawdown_comparison.csv")
        if args.output_plot:
            render_series_comparison_plot(
                result.nav_comparison,
                output_path=outdir / "nav_comparison.png",
                title="ATF trend sensitivity",
                ylabel="NAV",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
