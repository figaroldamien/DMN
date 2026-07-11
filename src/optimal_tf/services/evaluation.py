from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import pandas as pd

from optimal_tf.allocation import supported_strategies
from optimal_tf.rebalance import supported_rebalance_frequencies
from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_yf
from optimal_tf.data_quality import load_filtered_prices_for_universe
from optimal_tf.evaluation import evaluate_portfolio
from optimal_tf.scripts.common import (
    build_scenario_highlights,
    build_scenario_summary,
    eigenvalue_rows,
    matrix_benchmark_rows,
    matrix_sample,
    merge_common_overrides,
    parse_windows,
    reference_pipe_row,
    render_scenario_summary_text,
    render_scree_overview,
    resolve_target_dates,
    resolve_window_estimation_cfg,
    validate_methods,
    validate_strategies,
    write_matrix_pivots,
)
from trading_core.backtest.comparison import build_drawdown_comparison
from trading_core.market import get_universe_benchmark
from trading_core.reporting import (
    cumulative_nav,
    equal_weight_rebalanced_benchmark,
    evaluation_metrics,
    render_series_comparison_plot,
    single_asset_buy_and_hold_benchmark,
)
from trading_core.risk import supported_cleaning_methods

from .io import ensure_output_dir, write_json, write_quality_artifacts, write_request_json
from .models import (
    HyperparameterTuningRequest,
    HyperparameterTuningResult,
    RunArtifacts,
    VaryCleaningRequest,
    VaryCleaningResult,
    VaryFrequencyRequest,
    VaryFrequencyResult,
    VaryStrategyRequest,
    VaryStrategyResult,
    VaryWindowRequest,
    VaryWindowResult,
)


@dataclass(frozen=True)
class _HyperparameterGridContext:
    universe_name: str
    prices: pd.DataFrame
    quality_report: dict[str, object]
    estimation: object
    backtest: object
    evaluation: object
    strategies: list[str]
    methods: list[str]
    windows: list[int]
    frequencies: list[str]


DEFAULT_HYPERPARAMETER_WINDOWS = [40, 60, 80, 120, 252, 504]


def _with_cleaning_overrides(estimation, *, method: str | None = None, linear_shrinkage: float | None = None):
    updates = {}
    if method is not None:
        updates["cleaning_method"] = method
    if linear_shrinkage is not None:
        updates["linear_shrinkage"] = float(linear_shrinkage)
    if not updates:
        return estimation
    return replace(estimation, **updates)


def _build_primary_benchmark_bundle(
    *,
    universe_name: str,
    prices: pd.DataFrame,
    start: str,
    target_index: pd.Index,
    max_abs_return: float | None,
) -> tuple[str | None, dict[str, float | int] | None, pd.Series, pd.Series]:
    if len(target_index) == 0:
        empty = pd.Series(dtype=float)
        return None, None, empty, empty
    benchmark = get_universe_benchmark(universe_name)
    if benchmark and benchmark.get("ticker"):
        try:
            benchmark_prices = load_prices_yf([str(benchmark["ticker"])], start=start)
            benchmark_returns = single_asset_buy_and_hold_benchmark(
                benchmark_prices,
                max_abs_return=max_abs_return,
            )
            benchmark_label = str(benchmark.get("name") or benchmark.get("ticker"))
        except Exception:
            benchmark_returns = equal_weight_rebalanced_benchmark(prices, max_abs_return=max_abs_return)
            benchmark_label = "universe equal-weight index"
    else:
        benchmark_returns = equal_weight_rebalanced_benchmark(prices, max_abs_return=max_abs_return)
        benchmark_label = "universe equal-weight index"
    aligned_returns = benchmark_returns.reindex(pd.Index(target_index)).ffill().fillna(0.0)
    zero_turnover = pd.Series(0.0, index=aligned_returns.index, dtype=float)
    zero_costs = pd.Series(0.0, index=aligned_returns.index, dtype=float)
    benchmark_summary = evaluation_metrics(aligned_returns, zero_turnover, zero_costs, num_rebalances=0)
    benchmark_nav = cumulative_nav(aligned_returns).reindex(pd.Index(target_index)).ffill()
    benchmark_drawdown = benchmark_nav.divide(benchmark_nav.cummax()).subtract(1.0)
    benchmark_payload = dict(benchmark_summary.__dict__)
    benchmark_payload["final_nav"] = float(benchmark_nav.iloc[-1]) if len(benchmark_nav) else 1.0
    return benchmark_label, benchmark_payload, benchmark_nav, benchmark_drawdown


def _write_scenario_outputs(
    *,
    outdir: Path | None,
    matrix_frame: pd.DataFrame,
    strategy_frame: pd.DataFrame,
    nav_frame: pd.DataFrame,
    drawdown_frame: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    plots: dict[str, Path],
    summary_payload: dict,
    request,
    extra_frames: dict[str, pd.DataFrame] | None = None,
) -> RunArtifacts:
    files: dict[str, Path] = {}
    if outdir is None:
        return RunArtifacts(root_dir=None, files=files)
    matrix_frame.to_csv(outdir / 'matrix_benchmark.csv', index=False)
    strategy_frame.to_csv(outdir / 'strategy_benchmark.csv', index=False)
    nav_frame.to_csv(outdir / 'nav_comparison.csv')
    drawdown_frame.to_csv(outdir / 'drawdown_comparison.csv')
    scenario_summary.to_csv(outdir / 'scenario_summary.csv', index=False)
    files.update({
        'matrix': outdir / 'matrix_benchmark.csv',
        'strategy': outdir / 'strategy_benchmark.csv',
        'nav': outdir / 'nav_comparison.csv',
        'drawdown': outdir / 'drawdown_comparison.csv',
        'scenario_summary': outdir / 'scenario_summary.csv',
    })
    if extra_frames:
        for name, frame in extra_frames.items():
            path = outdir / f'{name}.csv'
            frame.to_csv(path, index=False)
            files[name] = path
    for key, value in plots.items():
        files[key] = value
    req = write_request_json(outdir, request)
    if req is not None:
        files['request'] = req
    write_json(outdir, 'summary.json', summary_payload)
    files['summary'] = outdir / 'summary.json'
    files.update(write_quality_artifacts(outdir, summary_payload.get('quality_report')))
    return RunArtifacts(root_dir=outdir, files=files)


def _resolve_hyperparameter_context(request: HyperparameterTuningRequest) -> _HyperparameterGridContext:
    universe, estimation, backtest, allocation, evaluation, compare, output = load_config(request.config_path)
    del allocation, compare, output
    universe, estimation, backtest, evaluation = merge_common_overrides(
        universe,
        estimation,
        backtest,
        evaluation,
        type('Args', (), {
            'universe': request.universe,
            'start': request.start,
            'rebalance_frequency': request.rebalance_frequency,
            'evaluation_start': request.evaluation_start,
            'evaluation_end': request.evaluation_end,
        })(),
    )
    if request.linear_shrinkage is not None:
        estimation = replace(estimation, linear_shrinkage=float(request.linear_shrinkage))
    if request.weight_smoothing_alpha is not None:
        backtest = replace(backtest, weight_smoothing_alpha=float(request.weight_smoothing_alpha))
    strategies = request.strategies or supported_strategies()
    validate_strategies(strategies)
    methods = request.methods or list(supported_cleaning_methods())
    validate_methods(methods)
    windows = request.windows or list(DEFAULT_HYPERPARAMETER_WINDOWS)
    windows = parse_windows(','.join(str(item) for item in windows))
    frequencies = request.frequencies or [evaluation.rebalance_frequency]
    allowed_frequencies = set(supported_rebalance_frequencies())
    invalid_frequencies = [item for item in frequencies if item not in allowed_frequencies]
    if invalid_frequencies:
        raise ValueError(f"Unknown rebalance frequencies {invalid_frequencies}. Allowed values: {supported_rebalance_frequencies()}")
    prices, quality_report = load_filtered_prices_for_universe(
        universe,
        evaluation_start=evaluation.evaluation_start,
        refresh_policy=request.refresh_policy,
    )
    return _HyperparameterGridContext(
        universe_name=universe.name,
        prices=prices,
        quality_report=asdict(quality_report),
        estimation=estimation,
        backtest=backtest,
        evaluation=evaluation,
        strategies=strategies,
        methods=methods,
        windows=windows,
        frequencies=list(frequencies),
    )


def _run_hyperparameter_grid(request: HyperparameterTuningRequest) -> tuple[_HyperparameterGridContext, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    context = _resolve_hyperparameter_context(request)
    num_assets = int(context.prices.shape[1])
    rows: list[dict[str, float | int | str]] = []
    skipped_rows: list[dict[str, float | int | str]] = []

    for strategy, method, window, frequency in product(context.strategies, context.methods, context.windows, context.frequencies):
        if method.startswith('rie') and int(window) <= num_assets:
            skipped_rows.append({
                'strategy': strategy,
                'method': method,
                'covariance_window': int(window),
                'rebalance_frequency': frequency,
                'num_assets': num_assets,
                'reason': 'rie_requires_window_gt_num_assets',
            })
            continue
        window_estimation = resolve_window_estimation_cfg(
            _with_cleaning_overrides(
                context.estimation,
                method=method,
                linear_shrinkage=request.linear_shrinkage,
            ),
            window,
            min_periods_mode=request.min_periods_mode,
        )
        result = evaluate_portfolio(
            context.prices,
            window_estimation,
            context.backtest,
            replace(context.evaluation, strategy=strategy, rebalance_frequency=frequency),
        )
        payload = dict(result.summary.__dict__)
        payload.update({
            'strategy': strategy,
            'method': method,
            'covariance_window': int(window),
            'rebalance_frequency': frequency,
            'covariance_min_periods': int(window_estimation.covariance_min_periods),
            'final_nav': float(cumulative_nav(result.daily_returns_net).iloc[-1]) if len(result.daily_returns_net) else 1.0,
        })
        rows.append(payload)

    results_table = pd.DataFrame(rows)
    skipped_configs = pd.DataFrame(skipped_rows)
    if not results_table.empty:
        sort_cols = [col for col in ['sharpe', 'total_return'] if col in results_table.columns]
        if sort_cols:
            results_table = results_table.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    highlights: dict[str, str] = {}
    if not results_table.empty:
        def _combo_label(frame_row) -> str:
            return (
                f"{frame_row['strategy']} | {frame_row['method']} | "
                f"window={int(frame_row['covariance_window'])} | freq={frame_row['rebalance_frequency']}"
            )

        if 'sharpe' in results_table.columns:
            row = results_table.sort_values('sharpe', ascending=False).iloc[0]
            highlights['best_sharpe'] = f"{_combo_label(row)} ({row['sharpe']:.4f})"
        if 'total_return' in results_table.columns:
            row = results_table.sort_values('total_return', ascending=False).iloc[0]
            highlights['best_total_return'] = f"{_combo_label(row)} ({row['total_return']:.4f})"
        if 'mdd' in results_table.columns:
            row = results_table.sort_values('mdd', ascending=False).iloc[0]
            highlights['lowest_drawdown'] = f"{_combo_label(row)} ({row['mdd']:.4f})"
    if not skipped_configs.empty:
        highlights['skipped_configs'] = str(len(skipped_configs))

    return context, results_table, skipped_configs, highlights


def run_hyperparameter_tuning(request: HyperparameterTuningRequest) -> HyperparameterTuningResult:
    context, results_table, skipped_configs, highlights = _run_hyperparameter_grid(request)
    outdir = ensure_output_dir(request.output_dir or 'output/optimal_tf/evaluation/hyperparameter_tuning')
    files: dict[str, Path] = {}
    if outdir is not None:
        results_path = outdir / 'results_table.csv'
        results_table.to_csv(results_path, index=False)
        files['results_table'] = results_path
        if not skipped_configs.empty:
            skipped_path = outdir / 'skipped_configs.csv'
            skipped_configs.to_csv(skipped_path, index=False)
            files['skipped_configs'] = skipped_path
        req = write_request_json(outdir, request)
        if req is not None:
            files['request'] = req
        write_json(
            outdir,
            'summary.json',
            {
                'universe': context.universe_name,
                'strategies': context.strategies,
                'methods': context.methods,
                'windows': context.windows,
                'frequencies': context.frequencies,
                'num_assets': int(context.prices.shape[1]),
                'skipped_configs': int(len(skipped_configs)),
                'highlights': highlights,
                'quality_report': context.quality_report,
            },
        )
        files['summary'] = outdir / 'summary.json'
        files.update(write_quality_artifacts(outdir, context.quality_report))

    return HyperparameterTuningResult(
        request=request,
        universe=context.universe_name,
        results_table=results_table,
        skipped_configs=skipped_configs,
        highlights=highlights,
        quality_report=context.quality_report,
        artifacts=RunArtifacts(root_dir=outdir, files=files),
    )


def run_vary_cleaning(request: VaryCleaningRequest) -> VaryCleaningResult:
    base_request = HyperparameterTuningRequest(
        config_path=request.config_path,
        universe=request.universe,
        start=request.start,
        evaluation_start=request.evaluation_start,
        evaluation_end=request.evaluation_end,
        rebalance_frequency=request.rebalance_frequency,
        strategies=[request.strategy] if request.strategy is not None else [],
        methods=request.methods,
        linear_shrinkage=request.linear_shrinkage,
        windows=[request.window] if request.window is not None else [],
        weight_smoothing_alpha=request.weight_smoothing_alpha,
        output_dir=request.output_dir,
        refresh_policy=request.refresh_policy,
    )
    context, results_table, skipped_configs, highlights = _run_hyperparameter_grid(base_request)
    if len(context.methods) < 2:
        raise ValueError('run_vary_cleaning expects at least two cleaning methods.')
    if len(context.windows) != 1:
        raise ValueError('run_vary_cleaning expects exactly one covariance window in the backend grid.')
    strategy = context.strategies[0]

    matrix_date = None
    if request.matrix_date is not None:
        matrix_date = pd.Timestamp(request.matrix_date)
    else:
        target_dates = resolve_target_dates(context.prices, replace(context.evaluation, strategy=strategy))
        if len(target_dates) == 0:
            raise ValueError('No evaluation rebalance dates available for the benchmark window.')
        matrix_date = target_dates[-1]

    cleaning_estimation = resolve_window_estimation_cfg(
        _with_cleaning_overrides(
            context.estimation,
            linear_shrinkage=request.linear_shrinkage,
        ),
        context.windows[0],
        min_periods_mode='clamp',
    )
    matrix_rows = matrix_benchmark_rows(context.prices, cleaning_estimation, context.methods, matrix_date)
    reference_pipe = reference_pipe_row(context.prices, cleaning_estimation, matrix_date)
    if reference_pipe is not None:
        matrix_rows.append(reference_pipe)

    empirical_corr, sample_size, sample_frame = matrix_sample(context.prices, cleaning_estimation, matrix_date)
    scree_rows = eigenvalue_rows(
        empirical_corr,
        sample_size,
        sample_frame,
        cleaning_estimation,
        context.methods,
        matrix_date=matrix_date,
    )

    strategy_frame = (
        results_table.loc[results_table['strategy'] == strategy]
        .copy()
        .sort_values('sharpe', ascending=False)
        .reset_index(drop=True)
    )
    method_results = {
        method: evaluate_portfolio(
            context.prices,
            _with_cleaning_overrides(
                cleaning_estimation,
                method=method,
                linear_shrinkage=request.linear_shrinkage,
            ),
            context.backtest,
            replace(context.evaluation, strategy=strategy),
        )
        for method in context.methods
        if method in strategy_frame['method'].tolist()
    }
    matrix_frame = pd.DataFrame(matrix_rows)
    scree_frame = pd.DataFrame(scree_rows).sort_values(['covariance_window', 'method', 'rank']).reset_index(drop=True)
    nav_frame = pd.concat(
        [cumulative_nav(result.daily_returns_net).rename(method) for method, result in method_results.items()],
        axis=1,
    ).sort_index().ffill()
    drawdown_frame = build_drawdown_comparison(nav_frame)
    benchmark_label, benchmark_summary, benchmark_nav, benchmark_drawdown = _build_primary_benchmark_bundle(
        universe_name=context.universe_name,
        prices=context.prices,
        start=request.start or context.prices.index.min().date().isoformat(),
        target_index=nav_frame.index,
        max_abs_return=getattr(context.estimation, "max_abs_return", None),
    )
    scenario_summary = build_scenario_summary(strategy_frame, 'method')
    scenario_highlights = build_scenario_highlights(strategy_frame, 'method')
    if not skipped_configs.empty:
        scenario_highlights['skipped_configs'] = str(len(skipped_configs))

    outdir = ensure_output_dir(request.output_dir or 'output/optimal_tf/evaluation/vary_cleaning')
    plots: dict[str, Path] = {}
    if outdir is not None:
        plots_dir = outdir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plots['nav_plot'] = render_series_comparison_plot(nav_frame, plots_dir / 'nav_comparison.png', title=f'Cleaning method NAV comparison ({strategy}, {context.universe_name})', ylabel='Cumulative value')
        plots['drawdown_plot'] = render_series_comparison_plot(drawdown_frame, plots_dir / 'drawdown_comparison.png', title=f'Cleaning method drawdown comparison ({strategy}, {context.universe_name})', ylabel='Drawdown')
        plots['scree_plot'] = render_scree_overview(scree_frame, outdir / 'cleaning_scree_overview.png', log_scale=request.log_scale)

    summary_payload = {
        'universe': context.universe_name,
        'scenario_key': 'method',
        'strategy': strategy,
        'methods': context.methods,
        'covariance_window': int(context.windows[0]),
        'matrix_date': matrix_date.strftime('%Y-%m-%d'),
        'highlights': scenario_highlights,
        'scenario_summary_text': render_scenario_summary_text(strategy_frame, 'method'),
        'quality_report': context.quality_report,
    }
    artifacts = _write_scenario_outputs(
        outdir=outdir,
        matrix_frame=matrix_frame,
        strategy_frame=strategy_frame,
        nav_frame=nav_frame,
        drawdown_frame=drawdown_frame,
        scenario_summary=scenario_summary,
        plots=plots,
        summary_payload=summary_payload,
        request=request,
        extra_frames={'cleaning_scree': scree_frame},
    )
    return VaryCleaningResult(
        request=request,
        universe=context.universe_name,
        scenario_key='method',
        scenario_summary=scenario_summary,
        strategy_benchmark=strategy_frame,
        matrix_benchmark=matrix_frame,
        nav_comparison=nav_frame,
        drawdown_comparison=drawdown_frame,
        benchmark_label=benchmark_label,
        benchmark_summary=benchmark_summary,
        benchmark_nav=benchmark_nav,
        benchmark_drawdown=benchmark_drawdown,
        highlights=scenario_highlights,
        quality_report=context.quality_report,
        artifacts=artifacts,
        scree_frame=scree_frame,
    )


def run_vary_window(request: VaryWindowRequest) -> VaryWindowResult:
    base_request = HyperparameterTuningRequest(
        config_path=request.config_path,
        universe=request.universe,
        start=request.start,
        evaluation_start=request.evaluation_start,
        evaluation_end=request.evaluation_end,
        rebalance_frequency=request.rebalance_frequency,
        strategies=[request.strategy] if request.strategy is not None else [],
        methods=[request.method] if request.method is not None else [],
        linear_shrinkage=request.linear_shrinkage,
        windows=request.windows,
        weight_smoothing_alpha=request.weight_smoothing_alpha,
        output_dir=request.output_dir,
        min_periods_mode=request.min_periods_mode,
        refresh_policy=request.refresh_policy,
    )
    context, results_table, skipped_configs, highlights = _run_hyperparameter_grid(base_request)
    if len(context.windows) < 2:
        raise ValueError('run_vary_window expects at least two windows.')
    if len(context.methods) != 1:
        raise ValueError('run_vary_window expects exactly one cleaning method in the backend grid.')
    if len(context.strategies) != 1:
        raise ValueError('run_vary_window expects exactly one strategy in the backend grid.')
    strategy = context.strategies[0]
    method = context.methods[0]

    target_dates = resolve_target_dates(context.prices, replace(context.evaluation, strategy=strategy))
    if len(target_dates) == 0:
        raise ValueError('No evaluation rebalance dates available for the benchmark window.')
    matrix_date = pd.Timestamp(request.matrix_date) if request.matrix_date is not None else target_dates[-1]

    matrix_rows = []
    scree_rows = []
    nav_series = []
    strategy_frame = (
        results_table.loc[(results_table['strategy'] == strategy) & (results_table['method'] == method)]
        .copy()
        .sort_values(['covariance_window'])
        .reset_index(drop=True)
    )

    for window in context.windows:
        window_estimation = resolve_window_estimation_cfg(
            _with_cleaning_overrides(
                context.estimation,
                method=method,
                linear_shrinkage=request.linear_shrinkage,
            ),
            window,
            min_periods_mode=request.min_periods_mode,
        )
        for row in matrix_benchmark_rows(context.prices, window_estimation, [method], matrix_date):
            row['covariance_window'] = int(window)
            row['covariance_min_periods'] = int(window_estimation.covariance_min_periods)
            matrix_rows.append(row)
        result = evaluate_portfolio(context.prices, window_estimation, context.backtest, replace(context.evaluation, strategy=strategy))
        nav_series.append(cumulative_nav(result.daily_returns_net).rename(f'window_{window}'))
        empirical_corr, sample_size, sample_frame = matrix_sample(context.prices, window_estimation, matrix_date)
        scree_rows.extend(eigenvalue_rows(empirical_corr, sample_size, sample_frame, window_estimation, [method], matrix_date=matrix_date))

    matrix_frame = pd.DataFrame(matrix_rows).sort_values(['covariance_window', 'method']).reset_index(drop=True)
    scree_frame = pd.DataFrame(scree_rows).sort_values(['covariance_window', 'method', 'rank']).reset_index(drop=True)
    nav_frame = pd.concat(nav_series, axis=1).sort_index().ffill()
    drawdown_frame = build_drawdown_comparison(nav_frame)
    benchmark_label, benchmark_summary, benchmark_nav, benchmark_drawdown = _build_primary_benchmark_bundle(
        universe_name=context.universe_name,
        prices=context.prices,
        start=request.start or context.prices.index.min().date().isoformat(),
        target_index=nav_frame.index,
        max_abs_return=getattr(context.estimation, "max_abs_return", None),
    )
    scenario_summary = build_scenario_summary(strategy_frame, 'covariance_window')
    scenario_highlights = build_scenario_highlights(strategy_frame, 'covariance_window')
    if not skipped_configs.empty:
        scenario_highlights['skipped_configs'] = str(len(skipped_configs))

    outdir = ensure_output_dir(request.output_dir or 'output/optimal_tf/evaluation/vary_window')
    plots: dict[str, Path] = {}
    if outdir is not None:
        plots_dir = outdir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plots['scree_plot'] = render_scree_overview(scree_frame, outdir / 'window_scree_overview.png', log_scale=request.log_scale)
        plots['nav_plot'] = render_series_comparison_plot(nav_frame, plots_dir / 'nav_comparison.png', title=f'Covariance window NAV comparison ({strategy}, {method}, {context.universe_name})', ylabel='Cumulative value')
        plots['drawdown_plot'] = render_series_comparison_plot(drawdown_frame, plots_dir / 'drawdown_comparison.png', title=f'Covariance window drawdown comparison ({strategy}, {method}, {context.universe_name})', ylabel='Drawdown')
        write_matrix_pivots(matrix_frame, outdir)

    summary_payload = {
        'universe': context.universe_name,
        'scenario_key': 'covariance_window',
        'strategy': strategy,
        'method': method,
        'windows': context.windows,
        'matrix_date': matrix_date.strftime('%Y-%m-%d'),
        'highlights': scenario_highlights,
        'scenario_summary_text': render_scenario_summary_text(strategy_frame, 'covariance_window'),
        'quality_report': context.quality_report,
    }
    artifacts = _write_scenario_outputs(
        outdir=outdir,
        matrix_frame=matrix_frame,
        strategy_frame=strategy_frame,
        nav_frame=nav_frame,
        drawdown_frame=drawdown_frame,
        scenario_summary=scenario_summary,
        plots=plots,
        summary_payload=summary_payload,
        request=request,
        extra_frames={'window_scree': scree_frame},
    )
    return VaryWindowResult(
        request=request,
        universe=context.universe_name,
        scenario_key='covariance_window',
        scenario_summary=scenario_summary,
        strategy_benchmark=strategy_frame,
        matrix_benchmark=matrix_frame,
        nav_comparison=nav_frame,
        drawdown_comparison=drawdown_frame,
        benchmark_label=benchmark_label,
        benchmark_summary=benchmark_summary,
        benchmark_nav=benchmark_nav,
        benchmark_drawdown=benchmark_drawdown,
        highlights=scenario_highlights,
        quality_report=context.quality_report,
        artifacts=artifacts,
        scree_frame=scree_frame,
    )


def run_vary_frequency(request: VaryFrequencyRequest) -> VaryFrequencyResult:
    base_request = HyperparameterTuningRequest(
        config_path=request.config_path,
        universe=request.universe,
        start=request.start,
        evaluation_start=request.evaluation_start,
        evaluation_end=request.evaluation_end,
        frequencies=request.frequencies,
        strategies=[request.strategy] if request.strategy is not None else [],
        methods=[request.method] if request.method is not None else [],
        linear_shrinkage=request.linear_shrinkage,
        windows=[request.window] if request.window is not None else [],
        weight_smoothing_alpha=request.weight_smoothing_alpha,
        output_dir=request.output_dir,
        min_periods_mode=request.min_periods_mode,
        refresh_policy=request.refresh_policy,
    )
    context, results_table, skipped_configs, highlights = _run_hyperparameter_grid(base_request)
    if len(context.frequencies) < 2:
        raise ValueError('run_vary_frequency expects at least two rebalance frequencies.')
    if len(context.methods) != 1:
        raise ValueError('run_vary_frequency expects exactly one cleaning method in the backend grid.')
    if len(context.strategies) != 1:
        raise ValueError('run_vary_frequency expects exactly one strategy in the backend grid.')
    if len(context.windows) != 1:
        raise ValueError('run_vary_frequency expects exactly one covariance window in the backend grid.')

    strategy = context.strategies[0]
    method = context.methods[0]
    window = context.windows[0]
    estimation = resolve_window_estimation_cfg(
        _with_cleaning_overrides(
            context.estimation,
            method=method,
            linear_shrinkage=request.linear_shrinkage,
        ),
        window,
        min_periods_mode=request.min_periods_mode,
    )

    target_dates = resolve_target_dates(context.prices, replace(context.evaluation, strategy=strategy, rebalance_frequency=context.frequencies[0]))
    if len(target_dates) == 0:
        raise ValueError('No evaluation rebalance dates available for the benchmark window.')
    matrix_date = pd.Timestamp(request.matrix_date) if request.matrix_date is not None else target_dates[-1]

    matrix_rows = []
    nav_series = []
    strategy_frame = (
        results_table.loc[
            (results_table['strategy'] == strategy)
            & (results_table['method'] == method)
            & (results_table['covariance_window'] == int(window))
        ]
        .copy()
        .sort_values(['rebalance_frequency'])
        .reset_index(drop=True)
    )

    for frequency in context.frequencies:
        for row in matrix_benchmark_rows(context.prices, estimation, [method], matrix_date):
            row['rebalance_frequency'] = frequency
            matrix_rows.append(row)
        result = evaluate_portfolio(
            context.prices,
            estimation,
            context.backtest,
            replace(context.evaluation, strategy=strategy, rebalance_frequency=frequency),
        )
        nav_series.append(cumulative_nav(result.daily_returns_net).rename(frequency))

    matrix_frame = pd.DataFrame(matrix_rows).sort_values(['rebalance_frequency', 'method']).reset_index(drop=True)
    nav_frame = pd.concat(nav_series, axis=1).sort_index().ffill()
    drawdown_frame = build_drawdown_comparison(nav_frame)
    benchmark_label, benchmark_summary, benchmark_nav, benchmark_drawdown = _build_primary_benchmark_bundle(
        universe_name=context.universe_name,
        prices=context.prices,
        start=request.start or context.prices.index.min().date().isoformat(),
        target_index=nav_frame.index,
        max_abs_return=getattr(context.estimation, "max_abs_return", None),
    )
    scenario_summary = build_scenario_summary(strategy_frame, 'rebalance_frequency')
    scenario_highlights = build_scenario_highlights(strategy_frame, 'rebalance_frequency')
    if not skipped_configs.empty:
        scenario_highlights['skipped_configs'] = str(len(skipped_configs))

    outdir = ensure_output_dir(request.output_dir or 'output/optimal_tf/evaluation/vary_frequency')
    plots: dict[str, Path] = {}
    if outdir is not None:
        plots_dir = outdir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plots['nav_plot'] = render_series_comparison_plot(
            nav_frame,
            plots_dir / 'nav_comparison.png',
            title=f'Rebalance frequency NAV comparison ({strategy}, {method}, window={window}, {context.universe_name})',
            ylabel='Cumulative value',
        )
        plots['drawdown_plot'] = render_series_comparison_plot(
            drawdown_frame,
            plots_dir / 'drawdown_comparison.png',
            title=f'Rebalance frequency drawdown comparison ({strategy}, {method}, window={window}, {context.universe_name})',
            ylabel='Drawdown',
        )

    summary_payload = {
        'universe': context.universe_name,
        'scenario_key': 'rebalance_frequency',
        'strategy': strategy,
        'method': method,
        'covariance_window': int(window),
        'frequencies': context.frequencies,
        'matrix_date': matrix_date.strftime('%Y-%m-%d'),
        'highlights': scenario_highlights,
        'scenario_summary_text': render_scenario_summary_text(strategy_frame, 'rebalance_frequency'),
        'quality_report': context.quality_report,
    }
    artifacts = _write_scenario_outputs(
        outdir=outdir,
        matrix_frame=matrix_frame,
        strategy_frame=strategy_frame,
        nav_frame=nav_frame,
        drawdown_frame=drawdown_frame,
        scenario_summary=scenario_summary,
        plots=plots,
        summary_payload=summary_payload,
        request=request,
    )
    return VaryFrequencyResult(
        request=request,
        universe=context.universe_name,
        scenario_key='rebalance_frequency',
        scenario_summary=scenario_summary,
        strategy_benchmark=strategy_frame,
        matrix_benchmark=matrix_frame,
        nav_comparison=nav_frame,
        drawdown_comparison=drawdown_frame,
        benchmark_label=benchmark_label,
        benchmark_summary=benchmark_summary,
        benchmark_nav=benchmark_nav,
        benchmark_drawdown=benchmark_drawdown,
        highlights=scenario_highlights,
        quality_report=context.quality_report,
        artifacts=artifacts,
    )


def run_vary_strategy(request: VaryStrategyRequest) -> VaryStrategyResult:
    base_request = HyperparameterTuningRequest(
        config_path=request.config_path,
        universe=request.universe,
        start=request.start,
        evaluation_start=request.evaluation_start,
        evaluation_end=request.evaluation_end,
        rebalance_frequency=request.rebalance_frequency,
        strategies=request.strategies,
        methods=[request.method] if request.method is not None else [],
        linear_shrinkage=request.linear_shrinkage,
        windows=[request.window] if request.window is not None else [],
        weight_smoothing_alpha=request.weight_smoothing_alpha,
        output_dir=request.output_dir,
        min_periods_mode=request.min_periods_mode,
        refresh_policy=request.refresh_policy,
    )
    context, results_table, skipped_configs, highlights = _run_hyperparameter_grid(base_request)
    if len(context.strategies) < 2:
        raise ValueError('run_vary_strategy expects at least two strategies.')
    if len(context.methods) != 1:
        raise ValueError('run_vary_strategy expects exactly one cleaning method in the backend grid.')
    if len(context.windows) != 1:
        raise ValueError('run_vary_strategy expects exactly one covariance window in the backend grid.')
    method = context.methods[0]
    estimation = resolve_window_estimation_cfg(
        _with_cleaning_overrides(
            context.estimation,
            method=method,
            linear_shrinkage=request.linear_shrinkage,
        ),
        context.windows[0],
        min_periods_mode=request.min_periods_mode,
    )

    target_dates = resolve_target_dates(context.prices, context.evaluation)
    if len(target_dates) == 0:
        raise ValueError('No evaluation rebalance dates available for the benchmark window.')
    matrix_date = pd.Timestamp(request.matrix_date) if request.matrix_date is not None else target_dates[-1]

    matrix_frame = pd.DataFrame(matrix_benchmark_rows(context.prices, estimation, [method], matrix_date))
    strategy_frame = (
        results_table.loc[
            (results_table['method'] == method)
            & (results_table['covariance_window'] == int(estimation.covariance_window or 0))
        ]
        .copy()
        .sort_values('sharpe', ascending=False)
        .reset_index(drop=True)
    )
    nav_series = []
    for strategy in context.strategies:
        result = evaluate_portfolio(context.prices, estimation, context.backtest, replace(context.evaluation, strategy=strategy))
        nav_series.append(cumulative_nav(result.daily_returns_net).rename(strategy))
    nav_frame = pd.concat(nav_series, axis=1).sort_index().ffill()
    drawdown_frame = build_drawdown_comparison(nav_frame)
    benchmark_label, benchmark_summary, benchmark_nav, benchmark_drawdown = _build_primary_benchmark_bundle(
        universe_name=context.universe_name,
        prices=context.prices,
        start=request.start or context.prices.index.min().date().isoformat(),
        target_index=nav_frame.index,
        max_abs_return=getattr(context.estimation, "max_abs_return", None),
    )
    scenario_summary = build_scenario_summary(strategy_frame, 'strategy')
    scenario_highlights = build_scenario_highlights(strategy_frame, 'strategy')
    if not skipped_configs.empty:
        scenario_highlights['skipped_configs'] = str(len(skipped_configs))

    outdir = ensure_output_dir(request.output_dir or 'output/optimal_tf/evaluation/vary_strategy')
    plots: dict[str, Path] = {}
    if outdir is not None:
        plots_dir = outdir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plots['nav_plot'] = render_series_comparison_plot(nav_frame, plots_dir / 'nav_comparison.png', title=f'Strategy NAV comparison ({method}, window={estimation.covariance_window}, {context.universe_name})', ylabel='Cumulative value')
        plots['drawdown_plot'] = render_series_comparison_plot(drawdown_frame, plots_dir / 'drawdown_comparison.png', title=f'Strategy drawdown comparison ({method}, window={estimation.covariance_window}, {context.universe_name})', ylabel='Drawdown')

    summary_payload = {
        'universe': context.universe_name,
        'scenario_key': 'strategy',
        'strategies': context.strategies,
        'method': method,
        'covariance_window': int(estimation.covariance_window or 0),
        'matrix_date': matrix_date.strftime('%Y-%m-%d'),
        'highlights': scenario_highlights,
        'scenario_summary_text': render_scenario_summary_text(strategy_frame, 'strategy'),
        'quality_report': context.quality_report,
    }
    artifacts = _write_scenario_outputs(
        outdir=outdir,
        matrix_frame=matrix_frame,
        strategy_frame=strategy_frame,
        nav_frame=nav_frame,
        drawdown_frame=drawdown_frame,
        scenario_summary=scenario_summary,
        plots=plots,
        summary_payload=summary_payload,
        request=request,
    )
    return VaryStrategyResult(
        request=request,
        universe=context.universe_name,
        scenario_key='strategy',
        scenario_summary=scenario_summary,
        strategy_benchmark=strategy_frame,
        matrix_benchmark=matrix_frame,
        nav_comparison=nav_frame,
        drawdown_comparison=drawdown_frame,
        benchmark_label=benchmark_label,
        benchmark_summary=benchmark_summary,
        benchmark_nav=benchmark_nav,
        benchmark_drawdown=benchmark_drawdown,
        highlights=scenario_highlights,
        quality_report=context.quality_report,
        artifacts=artifacts,
    )
