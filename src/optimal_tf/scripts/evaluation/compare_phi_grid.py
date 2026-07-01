from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.scripts.common import build_scenario_highlights, build_scenario_summary, merge_common_overrides
from optimal_tf.strategies_agnostic import compute_agnostic_panel
from trading_core.backtest import compare_strategies
from trading_core.backtest.engine import evaluate_portfolio
from trading_core.reporting import render_series_comparison_plot
from trading_core.risk import estimate_clean_correlation_panel

DEFAULT_OUTPUT_DIR = "output/optimal_tf/evaluation/phi_grid"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare agnostic phi-grid strategies for one signal family.")
    parser.add_argument("--config", type=str, default=str(Path("configs/optimal_tf.example.toml")))
    parser.add_argument("--universe", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--covariance-window", type=int, default=None)
    parser.add_argument("--covariance-min-periods", type=int, default=None)
    parser.add_argument("--evaluation-start", type=str, default=None)
    parser.add_argument("--evaluation-end", type=str, default=None)
    parser.add_argument("--rebalance-frequency", type=str, default=None)
    parser.add_argument("--signal-family", choices=("ones", "trend_ema"), default="ones")
    parser.add_argument("--phi-step", type=float, default=0.1)
    parser.add_argument("--refresh-policy", type=str, default="auto")
    parser.add_argument("--long-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-plot", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _phi_values(step: float) -> list[float]:
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"phi_step must lie in (0, 1], got {step}.")
    values: list[float] = []
    current = 0.0
    while current < 1.0 - 1e-12:
        values.append(round(current, 10))
        current += step
    values.append(1.0)
    unique = sorted({round(value, 10) for value in values})
    return unique


def _strategy_labels(signal_family: str, step: float) -> tuple[list[str], dict[str, float]]:
    values = _phi_values(step)
    prefix = "ARP" if signal_family == "ones" else "ATF"
    labels = [f"{prefix}_PHI_{value:.1f}" for value in values]
    mapping = dict(zip(labels, values))
    return labels, mapping


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels, phi_by_label = _strategy_labels(args.signal_family, args.phi_step)

    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(args.config)
    del allocation, compare, output
    universe, estimation, backtest, evaluation = merge_common_overrides(universe, estimation, backtest, evaluation, args)
    if args.long_only is not None:
        backtest = replace(backtest, long_only=bool(args.long_only))

    prices = load_prices_for_universe(universe.name, start=universe.start, refresh_policy=args.refresh_policy)

    def _evaluate_phi(prices_frame, estimation_cfg, backtest_cfg, evaluation_cfg):
        label = str(evaluation_cfg.strategy)
        phi = phi_by_label[label]
        return evaluate_portfolio(
            prices_frame,
            estimation_cfg,
            backtest_cfg,
            evaluation_cfg,
            compute_strategy_panel_fn=lambda p, e, strategy, **kwargs: compute_agnostic_panel(
                p,
                e,
                signal_model=args.signal_family,
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
        evaluate_portfolio_fn=_evaluate_phi,
    )

    summary = result.summary_table.copy()
    summary["phi"] = summary["strategy"].map(phi_by_label)
    cols = ["strategy", "phi", *[col for col in summary.columns if col not in {"strategy", "phi"}]]
    summary = summary.loc[:, cols].sort_values("phi").reset_index(drop=True)

    print(f"universe: {universe.name}")
    print(f"signal_family: {args.signal_family}")
    print(f"phi_values: {', '.join(f'{value:.1f}' for value in phi_by_label.values())}")
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
        summary.to_csv(outdir / f"scenario_summary_{args.signal_family}.csv", index=False)
        result.nav_comparison.to_csv(outdir / f"nav_comparison_{args.signal_family}.csv")
        result.drawdown_comparison.to_csv(outdir / f"drawdown_comparison_{args.signal_family}.csv")
        if args.output_plot:
            render_series_comparison_plot(
                result.nav_comparison,
                output_path=outdir / f"nav_comparison_{args.signal_family}.png",
                title=f"Phi grid comparison ({args.signal_family})",
                ylabel="NAV",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
