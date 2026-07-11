from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from optimal_tf.comparison import compare_strategies
from optimal_tf.config import AllocationConfig, BacktestConfig, EvaluationConfig, UniverseConfig
from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_yf
from optimal_tf.data_quality import load_filtered_prices_for_universe
from optimal_tf.evaluation import evaluate_portfolio
from optimal_tf.features import alpha_from_span, effective_span_from_alpha
from optimal_tf.rebalance import supported_rebalance_frequencies
from optimal_tf.strategies_agnostic import compute_agnostic_panel, supported_normalization_modes, supported_q_models, supported_signal_models
from optimal_tf.validation import validate_estimation_config
from trading_core.backtest.engine import evaluate_portfolio as _engine_evaluate_portfolio
from trading_core.market import get_universe_benchmark
from trading_core.reporting import (
    cumulative_nav,
    equal_weight_buy_and_hold_benchmark,
    equal_weight_rebalanced_benchmark,
    evaluation_metrics,
    render_evaluation_plot,
    render_series_comparison_plot,
    single_asset_buy_and_hold_benchmark,
    write_evaluation_outputs,
)
from trading_core.risk import estimate_clean_covariance_panel

from .io import ensure_output_dir, write_json, write_quality_artifacts, write_request_json
from .models import (
    AllocationRequest,
    AllocationResult,
    CompareRequest,
    CompareResult,
    RunArtifacts,
    StrategyTestbedRequest,
    StrategyTestbedResult,
    StandardEvaluationRequest,
    StandardEvaluationResult,
)
from ..allocation import compute_portfolio_strategy_state_at_date, compute_strategy_panel, supported_strategies


def _apply_common_overrides(
    universe: UniverseConfig,
    backtest: BacktestConfig,
    *,
    override_universe: str | None,
    override_start: str | None,
    override_long_only: bool | None,
) -> tuple[UniverseConfig, BacktestConfig]:
    if override_universe is not None:
        universe = UniverseConfig(name=override_universe, start=universe.start)
    if override_start is not None:
        universe = UniverseConfig(name=universe.name, start=override_start)
    if override_long_only is not None:
        backtest = replace(backtest, long_only=override_long_only)
    return universe, backtest


def _load_service_prices(
    universe: UniverseConfig,
    *,
    evaluation_start: str | None = None,
    refresh_policy: str,
) -> tuple[pd.DataFrame, dict]:
    prices, report = load_filtered_prices_for_universe(
        universe,
        evaluation_start=evaluation_start,
        refresh_policy=refresh_policy,
    )
    return prices, asdict(report)


def _apply_estimation_overrides(
    estimation,
    *,
    override_cleaning_method: str | None,
    override_linear_shrinkage: float | None,
    override_covariance_window: int | None,
    override_trend_alpha: float | None = None,
    override_trend_span: int | None = None,
):
    if override_cleaning_method is not None:
        estimation = replace(estimation, cleaning_method=override_cleaning_method)
    if override_linear_shrinkage is not None:
        estimation = replace(estimation, linear_shrinkage=float(override_linear_shrinkage))
    if override_covariance_window is not None:
        min_periods = min(estimation.covariance_min_periods, override_covariance_window)
        estimation = replace(
            estimation,
            covariance_window=override_covariance_window,
            covariance_min_periods=min_periods,
        )
    if override_trend_alpha is not None:
        estimation = replace(estimation, trend_alpha=float(override_trend_alpha))
    if override_trend_span is not None:
        estimation = replace(estimation, trend_span=int(override_trend_span))
    validate_estimation_config(estimation)
    return estimation


def _resolve_compare_strategies(raw: list[str], allocation: AllocationConfig, evaluation: EvaluationConfig, compare) -> list[str]:
    if raw:
        strategies = raw
    elif compare.strategies:
        strategies = list(compare.strategies)
    else:
        strategies = [evaluation.strategy or allocation.strategy]
    invalid = [item for item in strategies if item not in supported_strategies()]
    if invalid:
        raise ValueError(f"Unknown strategies {invalid}. Allowed values: {supported_strategies()}")
    if not strategies:
        raise ValueError("At least one strategy must be provided.")
    return strategies


def _load_primary_benchmark_returns(universe_name: str, prices: pd.DataFrame, *, start: str, max_abs_return: float | None) -> tuple[pd.Series, str, dict | None]:
    benchmark = get_universe_benchmark(universe_name)
    if benchmark and benchmark.get("ticker"):
        try:
            benchmark_prices = load_prices_yf([str(benchmark["ticker"])], start=start)
            benchmark_returns = single_asset_buy_and_hold_benchmark(benchmark_prices, max_abs_return=max_abs_return)
            label = str(benchmark.get("name") or benchmark.get("ticker"))
            return benchmark_returns, label, benchmark
        except Exception:
            pass
    benchmark_returns = equal_weight_rebalanced_benchmark(prices, max_abs_return=max_abs_return)
    return benchmark_returns, "universe equal-weight index", None


def _augment_comparison_with_benchmark(comparison, benchmark_returns: pd.Series, benchmark_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    benchmark_nav = cumulative_nav(benchmark_returns).reindex(comparison.nav_comparison.index).ffill()
    benchmark_drawdown = benchmark_nav.divide(benchmark_nav.cummax()).subtract(1.0)
    nav_comparison = comparison.nav_comparison.copy()
    drawdown_comparison = comparison.drawdown_comparison.copy()
    nav_comparison[benchmark_label] = benchmark_nav
    drawdown_comparison[benchmark_label] = benchmark_drawdown
    return nav_comparison, drawdown_comparison, benchmark_nav, benchmark_drawdown


def _augment_summary_with_benchmark(summary_table: pd.DataFrame, benchmark_returns: pd.Series, benchmark_label: str) -> pd.DataFrame:
    """Append the buy-and-hold benchmark row to a comparison summary table.

    `compare_strategies(...)` only knows about the strategy set it was asked to
    evaluate. The universe benchmark is resolved one layer above, so we add its
    metrics here to keep the Compare service aligned with Standard / Evaluation.
    """
    zero_turnover = pd.Series(0.0, index=benchmark_returns.index, dtype=float)
    zero_costs = pd.Series(0.0, index=benchmark_returns.index, dtype=float)
    benchmark_summary = evaluation_metrics(benchmark_returns, zero_turnover, zero_costs, num_rebalances=0)
    benchmark_row = {'strategy': benchmark_label, **benchmark_summary.__dict__}
    augmented = pd.concat([summary_table, pd.DataFrame([benchmark_row])], ignore_index=True)
    if 'sharpe' in augmented.columns:
        augmented = augmented.sort_values('sharpe', ascending=False, na_position='last')
    return augmented.reset_index(drop=True)


def _render_compare_plots(outdir: Path, comparison) -> tuple[Path, Path]:
    plots_dir = outdir / "comparison" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    nav_plot = render_series_comparison_plot(
        comparison.nav_comparison,
        plots_dir / "nav_comparison.png",
        title="Strategy NAV Comparison",
        ylabel="Cumulative value",
    )
    drawdown_plot = render_series_comparison_plot(
        comparison.drawdown_comparison,
        plots_dir / "drawdown_comparison.png",
        title="Strategy Drawdown Comparison",
        ylabel="Drawdown",
    )
    return nav_plot, drawdown_plot


def _validate_testbed_request(request: StrategyTestbedRequest) -> None:
    if request.strategy is not None and request.strategy not in supported_strategies():
        raise ValueError(f"Unknown strategy '{request.strategy}'. Allowed values: {supported_strategies()}")
    if request.q_model not in supported_q_models():
        raise ValueError(f"Unknown q_model '{request.q_model}'. Allowed values: {supported_q_models()}")
    if request.signal_model not in supported_signal_models():
        raise ValueError(f"Unknown signal_model '{request.signal_model}'. Allowed values: {supported_signal_models()}")
    if request.normalization not in supported_normalization_modes():
        raise ValueError(
            f"Unknown normalization mode '{request.normalization}'. Allowed values: {supported_normalization_modes()}"
        )
    if request.q_model != "phi_shrink_correlation" and float(request.phi) != 0.0:
        raise ValueError("phi is only supported with q_model='phi_shrink_correlation'.")
    if request.rebalance_frequency is not None and request.rebalance_frequency not in supported_rebalance_frequencies():
        raise ValueError(f"Unknown rebalance frequency '{request.rebalance_frequency}'.")


def _format_testbed_strategy_label(request: StrategyTestbedRequest) -> str:
    if request.strategy is not None:
        return str(request.strategy)
    return (
        "TESTBED"
        f"[signal={request.signal_model},q={request.q_model},phi={float(request.phi):.2f},"
        f"omega={float(request.omega):.2f},norm={request.normalization}]"
    )


def _resolve_testbed_trend_overrides(
    estimation,
    request: StrategyTestbedRequest,
) -> tuple[float | None, int | None]:
    """Make `trend_span` and `trend_alpha` overrides consistent for the testbed UI.

    The dashboard exposes both controls simultaneously, but the underlying EWMA
    resolver gives priority to `alpha` whenever both are present. We therefore
    infer user intent relative to the config defaults:
    - if only span changed, derive alpha from the new span
    - if only alpha changed, derive span from the new alpha
    - if both changed, alpha wins because the EWMA implementation uses it first
    """
    base_alpha = estimation.trend_alpha
    base_span = estimation.trend_span
    req_alpha = request.trend_alpha
    req_span = request.trend_span

    if req_alpha is None and req_span is None:
        return None, None

    alpha_changed = req_alpha is not None and req_alpha != base_alpha
    span_changed = req_span is not None and req_span != base_span

    if span_changed and not alpha_changed:
        return alpha_from_span(req_span), req_span
    if alpha_changed and not span_changed:
        return req_alpha, effective_span_from_alpha(req_alpha)
    if alpha_changed and span_changed:
        return req_alpha, effective_span_from_alpha(req_alpha)
    return req_alpha, req_span


def run_allocation(request: AllocationRequest) -> AllocationResult:
    universe, estimation, backtest, allocation, _, _, output = load_config(request.config_path)
    universe, backtest = _apply_common_overrides(
        universe,
        backtest,
        override_universe=request.universe,
        override_start=request.start,
        override_long_only=request.long_only,
    )
    estimation = _apply_estimation_overrides(
        estimation,
        override_cleaning_method=request.cleaning_method,
        override_linear_shrinkage=request.linear_shrinkage,
        override_covariance_window=request.covariance_window,
    )
    allocation = AllocationConfig(
        strategy=request.strategy or allocation.strategy,
        date=request.as_of_date if request.as_of_date is not None else allocation.date,
    )

    prices, quality_report = _load_service_prices(universe, refresh_policy=request.refresh_policy)
    allocation_date, state = compute_portfolio_strategy_state_at_date(
        prices,
        estimation,
        allocation.strategy,
        as_of_date=allocation.date,
        long_only=backtest.long_only,
    )

    outdir = ensure_output_dir(request.output_dir)
    files: dict[str, Path] = {}
    if outdir is not None:
        csv_path = outdir / "weights.csv"
        json_path = outdir / "weights.json"
        export = state.effective_weights.rename("weight").reset_index().rename(columns={"index": "ticker"})
        export.insert(0, "date", allocation_date.strftime("%Y-%m-%d"))
        export.insert(1, "strategy", allocation.strategy)
        export.insert(2, "universe", universe.name)
        export.to_csv(csv_path, index=False)
        payload = {
            "date": allocation_date.strftime("%Y-%m-%d"),
            "strategy": allocation.strategy,
            "universe": universe.name,
            "signal_scale": float(state.signal_scale),
            "base_weights": {str(k): float(v) for k, v in state.base_weights.items()},
            "weights": {str(k): float(v) for k, v in state.effective_weights.items()},
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        files["weights_csv"] = csv_path
        files["weights_json"] = json_path
        req_path = write_request_json(outdir, request)
        if req_path is not None:
            files["request"] = req_path
        files.update(write_quality_artifacts(outdir, quality_report))

    return AllocationResult(
        request=request,
        universe=universe.name,
        strategy=allocation.strategy,
        cleaning_method=estimation.cleaning_method,
        covariance_window=estimation.covariance_window,
        allocation_date=allocation_date,
        signal_scale=float(state.signal_scale),
        weights=state.effective_weights,
        base_weights=state.base_weights,
        quality_report=quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )


def run_evaluation(request: StandardEvaluationRequest) -> StandardEvaluationResult:
    universe, estimation, backtest, allocation, evaluation, _, output = load_config(request.config_path)
    universe, backtest = _apply_common_overrides(
        universe,
        backtest,
        override_universe=request.universe,
        override_start=request.start,
        override_long_only=request.long_only,
    )
    estimation = _apply_estimation_overrides(
        estimation,
        override_cleaning_method=request.cleaning_method,
        override_linear_shrinkage=request.linear_shrinkage,
        override_covariance_window=request.covariance_window,
    )
    if request.weight_smoothing_alpha is not None:
        backtest = replace(backtest, weight_smoothing_alpha=float(request.weight_smoothing_alpha))
    strategy = request.strategy or evaluation.strategy
    frequency = request.rebalance_frequency or evaluation.rebalance_frequency
    if request.rebalance_frequency is not None and request.rebalance_frequency not in supported_rebalance_frequencies():
        raise ValueError(f"Unknown rebalance frequency '{request.rebalance_frequency}'.")
    evaluation = EvaluationConfig(
        strategy=strategy,
        rebalance_frequency=frequency,
        evaluation_start=request.evaluation_start if request.evaluation_start is not None else evaluation.evaluation_start,
        evaluation_end=request.evaluation_end if request.evaluation_end is not None else evaluation.evaluation_end,
    )

    prices, quality_report = _load_service_prices(
        universe,
        evaluation_start=evaluation.evaluation_start,
        refresh_policy=request.refresh_policy,
    )
    result = evaluate_portfolio(prices, estimation, backtest, evaluation)
    benchmark_returns, benchmark_label, benchmark_metadata = _load_primary_benchmark_returns(
        universe.name,
        prices,
        start=universe.start,
        max_abs_return=estimation.max_abs_return,
    )
    benchmark_returns = benchmark_returns.reindex(result.daily_returns_net.index).ffill().fillna(0.0)
    buy_hold_returns = equal_weight_buy_and_hold_benchmark(prices, max_abs_return=estimation.max_abs_return).reindex(result.daily_returns_net.index).ffill().fillna(0.0)
    buy_hold_label = "equal-weight buy and hold"

    outdir = ensure_output_dir(request.output_dir or output.evaluation_dir)
    files: dict[str, Path] = {}
    if outdir is not None:
        write_evaluation_outputs(result, str(outdir))
        files["summary"] = outdir / "summary.json"
        files["weights_by_rebalance"] = outdir / "weights_by_rebalance.csv"
        files["daily_returns_net"] = outdir / "daily_returns_net.csv"
        benchmark_export = benchmark_returns.rename("return").to_frame()
        benchmark_export.to_csv(outdir / "benchmark_returns.csv")
        files["benchmark_returns"] = outdir / "benchmark_returns.csv"
        buy_hold_export = buy_hold_returns.rename("return").to_frame()
        buy_hold_export.to_csv(outdir / "buy_hold_returns.csv")
        files["buy_hold_returns"] = outdir / "buy_hold_returns.csv"
        if request.output_plot and (request.output_dir is not None or output.evaluation_plot):
            plot_path = render_evaluation_plot(
                result.daily_returns_net,
                benchmark_returns,
                buy_hold_returns,
                outdir / "performance.png",
                title=f"{evaluation.strategy} vs benchmark ({universe.name})",
                benchmark_label=benchmark_label,
                buy_hold_label=buy_hold_label,
            )
            files["plot"] = plot_path
        req_path = write_request_json(outdir, request)
        if req_path is not None:
            files["request"] = req_path
        files.update(write_quality_artifacts(outdir, quality_report))

    return StandardEvaluationResult(
        request=request,
        universe=universe.name,
        strategy=evaluation.strategy,
        cleaning_method=estimation.cleaning_method,
        covariance_window=estimation.covariance_window,
        rebalance_frequency=evaluation.rebalance_frequency,
        evaluation_result=result,
        benchmark_returns=benchmark_returns,
        benchmark_label=benchmark_label,
        benchmark_metadata=benchmark_metadata,
        buy_hold_returns=buy_hold_returns,
        buy_hold_label=buy_hold_label,
        quality_report=quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )


def run_strategy_testbed(request: StrategyTestbedRequest) -> StrategyTestbedResult:
    _validate_testbed_request(request)
    universe, estimation, backtest, allocation, evaluation, _, output = load_config(request.config_path)
    del allocation
    universe, backtest = _apply_common_overrides(
        universe,
        backtest,
        override_universe=request.universe,
        override_start=request.start,
        override_long_only=request.long_only,
    )
    trend_alpha_override, trend_span_override = _resolve_testbed_trend_overrides(estimation, request)
    estimation = _apply_estimation_overrides(
        estimation,
        override_cleaning_method=request.cleaning_method,
        override_linear_shrinkage=request.linear_shrinkage,
        override_covariance_window=request.covariance_window,
        override_trend_alpha=trend_alpha_override,
        override_trend_span=trend_span_override,
    )
    if request.weight_smoothing_alpha is not None:
        backtest = replace(backtest, weight_smoothing_alpha=float(request.weight_smoothing_alpha))
    evaluation = EvaluationConfig(
        strategy=_format_testbed_strategy_label(request),
        rebalance_frequency=request.rebalance_frequency or evaluation.rebalance_frequency,
        evaluation_start=request.evaluation_start if request.evaluation_start is not None else evaluation.evaluation_start,
        evaluation_end=request.evaluation_end if request.evaluation_end is not None else evaluation.evaluation_end,
    )
    prices, quality_report = _load_service_prices(
        universe,
        evaluation_start=evaluation.evaluation_start,
        refresh_policy=request.refresh_policy,
    )

    if request.strategy is not None:
        compute_strategy_panel_fn = compute_strategy_panel
    else:
        def _compute_testbed_panel(prices_frame, est_cfg, _strategy, *, long_only=False, target_dates=None, covariance_cache=None):
            return compute_agnostic_panel(
                prices_frame,
                est_cfg,
                signal_model=request.signal_model,  # type: ignore[arg-type]
                q_model=request.q_model,  # type: ignore[arg-type]
                phi=float(request.phi),
                omega=float(request.omega),
                normalization=request.normalization,  # type: ignore[arg-type]
                long_only=long_only,
                target_dates=target_dates,
                covariance_cache=covariance_cache,
            )
        compute_strategy_panel_fn = _compute_testbed_panel

    result = _engine_evaluate_portfolio(
        prices,
        estimation,
        backtest,
        evaluation,
        compute_strategy_panel_fn=compute_strategy_panel_fn,
        estimate_clean_covariance_panel_fn=estimate_clean_covariance_panel,
    )
    benchmark_returns, benchmark_label, benchmark_metadata = _load_primary_benchmark_returns(
        universe.name,
        prices,
        start=universe.start,
        max_abs_return=estimation.max_abs_return,
    )
    benchmark_returns = benchmark_returns.reindex(result.daily_returns_net.index).ffill().fillna(0.0)
    buy_hold_returns = equal_weight_buy_and_hold_benchmark(prices, max_abs_return=estimation.max_abs_return).reindex(
        result.daily_returns_net.index
    ).ffill().fillna(0.0)
    buy_hold_label = "equal-weight buy and hold"

    outdir = ensure_output_dir(request.output_dir or output.evaluation_dir)
    files: dict[str, Path] = {}
    if outdir is not None:
        write_evaluation_outputs(result, str(outdir))
        files["summary"] = outdir / "summary.json"
        files["weights_by_rebalance"] = outdir / "weights_by_rebalance.csv"
        files["daily_returns_net"] = outdir / "daily_returns_net.csv"
        benchmark_returns.rename("return").to_frame().to_csv(outdir / "benchmark_returns.csv")
        files["benchmark_returns"] = outdir / "benchmark_returns.csv"
        buy_hold_returns.rename("return").to_frame().to_csv(outdir / "buy_hold_returns.csv")
        files["buy_hold_returns"] = outdir / "buy_hold_returns.csv"
        if request.output_plot and (request.output_dir is not None or output.evaluation_plot):
            plot_path = render_evaluation_plot(
                result.daily_returns_net,
                benchmark_returns,
                buy_hold_returns,
                outdir / "performance.png",
                title=f"{evaluation.strategy} vs benchmark ({universe.name})",
                benchmark_label=benchmark_label,
                buy_hold_label=buy_hold_label,
            )
            files["plot"] = plot_path
        req_path = write_request_json(outdir, request)
        if req_path is not None:
            files["request"] = req_path
        files.update(write_quality_artifacts(outdir, quality_report))

    return StrategyTestbedResult(
        request=request,
        universe=universe.name,
        strategy_label=evaluation.strategy,
        cleaning_method=estimation.cleaning_method,
        covariance_window=estimation.covariance_window,
        rebalance_frequency=evaluation.rebalance_frequency,
        signal_model=request.signal_model,
        q_model=request.q_model,
        phi=float(request.phi),
        omega=float(request.omega),
        normalization=request.normalization,
        evaluation_result=result,
        benchmark_returns=benchmark_returns,
        benchmark_label=benchmark_label,
        benchmark_metadata=benchmark_metadata,
        buy_hold_returns=buy_hold_returns,
        buy_hold_label=buy_hold_label,
        quality_report=quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )


def run_compare(request: CompareRequest) -> CompareResult:
    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(request.config_path)
    universe, backtest = _apply_common_overrides(
        universe,
        backtest,
        override_universe=request.universe,
        override_start=request.start,
        override_long_only=request.long_only,
    )
    estimation = _apply_estimation_overrides(
        estimation,
        override_cleaning_method=request.cleaning_method,
        override_linear_shrinkage=request.linear_shrinkage,
        override_covariance_window=request.covariance_window,
    )
    if request.weight_smoothing_alpha is not None:
        backtest = replace(backtest, weight_smoothing_alpha=float(request.weight_smoothing_alpha))
    frequency = request.rebalance_frequency or evaluation.rebalance_frequency
    if request.rebalance_frequency is not None and request.rebalance_frequency not in supported_rebalance_frequencies():
        raise ValueError(f"Unknown rebalance frequency '{request.rebalance_frequency}'.")
    evaluation = EvaluationConfig(
        strategy=evaluation.strategy,
        rebalance_frequency=frequency,
        evaluation_start=request.evaluation_start if request.evaluation_start is not None else evaluation.evaluation_start,
        evaluation_end=request.evaluation_end if request.evaluation_end is not None else evaluation.evaluation_end,
    )
    strategies = _resolve_compare_strategies(request.strategies, allocation, evaluation, compare)
    prices, quality_report = _load_service_prices(
        universe,
        evaluation_start=evaluation.evaluation_start,
        refresh_policy=request.refresh_policy,
    )
    comparison = compare_strategies(prices, estimation, backtest, evaluation, strategies)
    benchmark_returns, benchmark_label, benchmark_metadata = _load_primary_benchmark_returns(
        universe.name,
        prices,
        start=universe.start,
        max_abs_return=estimation.max_abs_return,
    )
    benchmark_returns = benchmark_returns.reindex(comparison.nav_comparison.index).ffill().fillna(0.0)
    nav_comparison, drawdown_comparison, benchmark_nav, benchmark_drawdown = _augment_comparison_with_benchmark(
        comparison,
        benchmark_returns,
        benchmark_label,
    )
    summary_table = _augment_summary_with_benchmark(comparison.summary_table, benchmark_returns, benchmark_label)
    comparison = replace(
        comparison,
        summary_table=summary_table,
        nav_comparison=nav_comparison,
        drawdown_comparison=drawdown_comparison,
    )

    outdir = ensure_output_dir(request.output_dir or output.compare_dir, clean=request.clean_output_dir)
    files: dict[str, Path] = {}
    if outdir is not None:
        strategies_dir = outdir / "strategies"
        strategies_dir.mkdir(parents=True, exist_ok=True)
        for strategy_name, strategy_result in comparison.strategy_results.items():
            write_evaluation_outputs(strategy_result, str(strategies_dir / strategy_name))
        comp_dir = outdir / "comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)
        comparison.summary_table.to_csv(comp_dir / "summary_table.csv", index=False)
        nav_comparison.to_csv(comp_dir / "nav_comparison.csv")
        drawdown_comparison.to_csv(comp_dir / "drawdown_comparison.csv")
        files["summary_table"] = comp_dir / "summary_table.csv"
        files["nav"] = comp_dir / "nav_comparison.csv"
        files["drawdown"] = comp_dir / "drawdown_comparison.csv"
        write_json(
            outdir,
            "manifest.json",
            {
                "universe": universe.name,
                "start": universe.start,
                "rebalance_frequency": evaluation.rebalance_frequency,
                "evaluation_start": evaluation.evaluation_start,
                "evaluation_end": evaluation.evaluation_end,
                "long_only": backtest.long_only,
                "strategies": strategies,
                "cleaning_method": estimation.cleaning_method,
                "covariance_window": estimation.covariance_window,
            },
        )
        write_json(
            outdir,
            "inputs.json",
            {
                "universe": asdict(universe),
                "estimation": asdict(estimation),
                "backtest": asdict(backtest),
                "allocation": asdict(allocation),
                "evaluation": asdict(evaluation),
            },
        )
        if request.output_plot:
            nav_plot, drawdown_plot = _render_compare_plots(outdir, comparison)
            files["nav_plot"] = nav_plot
            files["drawdown_plot"] = drawdown_plot
        req_path = write_request_json(outdir, request)
        if req_path is not None:
            files["request"] = req_path
        files.update(write_quality_artifacts(outdir, quality_report))

    return CompareResult(
        request=request,
        universe=universe.name,
        strategies=strategies,
        cleaning_method=estimation.cleaning_method,
        covariance_window=estimation.covariance_window,
        rebalance_frequency=evaluation.rebalance_frequency,
        comparison=comparison,
        benchmark_label=benchmark_label,
        benchmark_metadata=benchmark_metadata,
        benchmark_nav=benchmark_nav,
        benchmark_drawdown=benchmark_drawdown,
        quality_report=quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )
