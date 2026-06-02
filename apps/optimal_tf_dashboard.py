from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st
from market_tickers_data import MARKET_TICKERS
from optimal_tf.allocation import supported_strategies
from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe
from optimal_tf.rebalance import resolve_rebalance_dates, supported_rebalance_frequencies
from optimal_tf.services import (
    AllocationRequest,
    CompareRequest,
    HyperparameterTuningRequest,
    InspectionSnapshotRequest,
    MarketSynthesisRequest,
    MarketSynthesisResult,
    StandardEvaluationRequest,
    VaryCleaningRequest,
    VaryFrequencyRequest,
    VaryStrategyRequest,
    VaryWindowRequest,
    run_allocation,
    run_compare,
    run_evaluation,
    run_hyperparameter_tuning,
    run_inspection_snapshot,
    run_market_synthesis,
    run_vary_cleaning,
    run_vary_frequency,
    run_vary_strategy,
    run_vary_window,
)
from trading_core.risk import marchenko_pastur_law, supported_cleaning_methods
from trading_core.reporting import evaluation_metrics
from trading_core.reporting.plots import plt

DEFAULT_CONFIG = 'configs/optimal_tf.example.toml'
UNIVERSE_OPTIONS = sorted(MARKET_TICKERS)
STRATEGY_OPTIONS = supported_strategies()
FREQUENCY_OPTIONS = supported_rebalance_frequencies()
CLEANING_OPTIONS = list(supported_cleaning_methods())

MARKET_SERVICES = {
    'Momentum synthesis': 'Cross-sectional momentum synthesis with optional sector and sub-sector aggregation.',
}

MOMENTUM_COLUMNS = ['annual', 'semiannual', 'quarterly', 'monthly', 'weekly', 'daily']
MOMENTUM_LABELS = {
    'annual': 'Annual',
    'semiannual': 'Semiannual',
    'quarterly': 'Quarterly',
    'monthly': 'Monthly',
    'weekly': 'Weekly',
    'daily': 'Daily',
}

MARKET_SORT_OPTIONS = {
    'hierarchy': 'Sector / sub-sector',
    'performance': 'Performance',
}

STANDARD_SERVICES = {
    'Allocation': 'Standard allocation of one strategy on one date.',
    'Evaluation': 'Standard backtest using the packaged configuration.',
    'Compare': 'Standard comparison of several strategies using the config defaults.',
}

TUNING_SERVICES = {
    'Vary cleaning': 'Compare correlation cleaning methods for one strategy.',
    'Vary window': 'Compare covariance lookback windows for one strategy and one cleaner.',
    'Vary strategy': 'Compare strategies for one cleaner and one window.',
    'Vary frequency': 'Compare rebalance frequencies for one strategy, one cleaner and one window.',
    'Hyperparameter tuning': 'Grid search over strategies, cleaning methods, covariance windows and rebalance frequencies.',
}

INSPECTION_SERVICES = {
    'Inspection snapshot': 'Inspect one dated allocation snapshot with matrices, spectra, features and weights.',
}

COMMON_COLUMN_CONFIG = {
    'strategy': st.column_config.TextColumn('Strategy', width='small'),
    'method': st.column_config.TextColumn('Cleaning', width='small'),
    'cleaning_method': st.column_config.TextColumn('Cleaning', width='small'),
    'covariance_window': st.column_config.NumberColumn('Corr.\nwindow', width='small', format='%d'),
    'window': st.column_config.NumberColumn('Window', width='small', format='%d'),
    'rebalance_frequency': st.column_config.TextColumn('Rebalance\nfrequency', width='small'),
    'sharpe': st.column_config.NumberColumn('Sharpe', width='small', format='%.3f'),
    'total_return': st.column_config.NumberColumn('Total\nreturn', width='small', format='%.3f'),
    'ann_return': st.column_config.NumberColumn('Ann.\nreturn', width='small', format='%.3f'),
    'ann_vol': st.column_config.NumberColumn('Ann.\nvol', width='small', format='%.3f'),
    'mdd': st.column_config.NumberColumn('Max\nDD', width='small', format='%.3f'),
    'avg_turnover': st.column_config.NumberColumn('Avg.\nturnover', width='small', format='%.3f'),
    'total_cost': st.column_config.NumberColumn('Total\ncost', width='small', format='%.3f'),
    'final_nav': st.column_config.NumberColumn('Final\nNAV', width='small', format='%.3f'),
    'signal_scale': st.column_config.NumberColumn('Signal\nscale', width='small', format='%.3f'),
    'allocation_date': st.column_config.TextColumn('Allocation\ndate', width='small'),
    'matrix_date': st.column_config.TextColumn('Matrix\ndate', width='small'),
    'rank': st.column_config.NumberColumn('Rank', width='small', format='%d'),
    'eigenvalue': st.column_config.NumberColumn('Eigen\nvalue', width='small', format='%.4f'),
    'variance_share': st.column_config.NumberColumn('Var.\nshare', width='small', format='%.4f'),
    'cumulative_variance_share': st.column_config.NumberColumn('Cum.\nvar', width='small', format='%.4f'),
    'num_assets': st.column_config.NumberColumn('Num\nassets', width='small', format='%d'),
    'sample_size': st.column_config.NumberColumn('Sample\nsize', width='small', format='%d'),
    'reason': st.column_config.TextColumn('Reason', width='medium'),
    'name': st.column_config.TextColumn('Artifact', width='medium'),
    'path': st.column_config.TextColumn('Path', width='large'),
    'ticker': st.column_config.TextColumn('Ticker', width='small'),
    'instrument': st.column_config.TextColumn('Instrument', width='medium'),
    'description': st.column_config.TextColumn('Name', width='medium'),
    'weight': st.column_config.NumberColumn('Weight', width='small', format='%.4f'),
    'base_weight': st.column_config.NumberColumn('Base\nweight', width='small', format='%.4f'),
    'effective_weight': st.column_config.NumberColumn('Eff.\nweight', width='small', format='%.4f'),
    'abs_effective_weight': st.column_config.NumberColumn('Abs eff.\nweight', width='small', format='%.4f'),
    'last_price': st.column_config.NumberColumn('Last\nprice', width='small', format='%.4f'),
    'last_return': st.column_config.NumberColumn('Last\nreturn', width='small', format='%.4f'),
    'ewma_vol': st.column_config.NumberColumn('EWMA\nvol', width='small', format='%.4f'),
    'z_return': st.column_config.NumberColumn('Z\nreturn', width='small', format='%.4f'),
    'trend_signal': st.column_config.NumberColumn('Trend\nsignal', width='small', format='%.4f'),
}

st.set_page_config(page_title='optimal_tf dashboard', layout='wide')
st.title('optimal_tf dashboard MVP')
st.caption('Pilotage interactif des services standard, tuning et inspection via l\'API Python.')


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return value.rename('value').reset_index().to_dict(orient='records')
    if isinstance(value, pd.DataFrame):
        return value.head(200).to_dict(orient='records')
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_defaults(config_path: str) -> tuple[dict[str, Any], str | None]:
    try:
        universe, estimation, backtest, allocation, evaluation, compare, output = load_config(config_path)
    except Exception as exc:  # pragma: no cover
        return {}, str(exc)
    defaults = {
        'universe': asdict(universe),
        'estimation': asdict(estimation),
        'backtest': asdict(backtest),
        'allocation': asdict(allocation),
        'evaluation': asdict(evaluation),
        'compare': asdict(compare),
        'output': asdict(output),
    }
    return defaults, None


def _selectbox_with_default(label: str, options: list[str], default_value: str) -> str:
    default_index = options.index(default_value) if default_value in options else 0
    return st.sidebar.selectbox(label, options, index=default_index)


def _multiselect_with_defaults(label: str, options: list[str], default_values: list[str]) -> list[str]:
    defaults = [item for item in default_values if item in options]
    return st.sidebar.multiselect(label, options, default=defaults)


def _parse_default_date(default_value: Any) -> pd.Timestamp:
    if default_value in (None, '', 'None'):
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(default_value).normalize()


def _date_input_value(label: str, default_value: Any, *, key: str) -> str:
    selected = st.sidebar.date_input(label, value=_parse_default_date(default_value).date(), key=key)
    return pd.Timestamp(selected).date().isoformat()


def _latest_or_date_input(label: str, default_value: Any, *, key_prefix: str, latest_default: bool = False) -> str | None:
    use_latest = st.sidebar.checkbox(f'Use latest available {label.lower()}', value=latest_default, key=f'{key_prefix}::latest')
    selected = st.sidebar.date_input(
        label,
        value=_parse_default_date(default_value).date(),
        key=f'{key_prefix}::date',
        disabled=use_latest,
    )
    if use_latest:
        return None
    return pd.Timestamp(selected).date().isoformat()


def _inspection_date_controls(config_defaults: dict[str, Any], *, universe: str, start: str) -> tuple[str | None, str]:
    freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
    rebalance_frequency = _selectbox_with_default('Inspection rebalance frequency', FREQUENCY_OPTIONS, freq_default)
    st.sidebar.caption(f'Default from config: {freq_default}')

    session_key = f'inspection_date::{universe}::{start or "default"}'
    if session_key not in st.session_state:
        try:
            prices = load_prices_for_universe(universe, start=start or None)
            default_date = pd.Timestamp(prices.index.max()).date() if len(prices.index) else pd.Timestamp.today().date()
        except Exception:
            default_date = pd.Timestamp.today().date()
        st.session_state[session_key] = default_date

    controls_left, controls_center, controls_right = st.sidebar.columns(3)
    if controls_left.button('Latest date', key=f'{session_key}:latest'):
        try:
            prices = load_prices_for_universe(universe, start=start or None)
            if len(prices.index):
                st.session_state[session_key] = pd.Timestamp(prices.index.max()).date()
        except Exception:
            pass
    if controls_center.button('Prev rebalance', key=f'{session_key}:prev'):
        try:
            prices = load_prices_for_universe(universe, start=start or None)
            rebalance_dates = resolve_rebalance_dates(prices.index, rebalance_frequency, start=start or None)
            current = pd.Timestamp(st.session_state[session_key])
            prev_dates = [pd.Timestamp(ts) for ts in rebalance_dates if pd.Timestamp(ts) < current]
            if prev_dates:
                st.session_state[session_key] = prev_dates[-1].date()
        except Exception:
            pass
    if controls_right.button('Next rebalance', key=f'{session_key}:next'):
        try:
            prices = load_prices_for_universe(universe, start=start or None)
            rebalance_dates = resolve_rebalance_dates(prices.index, rebalance_frequency, start=start or None)
            current = pd.Timestamp(st.session_state[session_key])
            next_dates = [pd.Timestamp(ts) for ts in rebalance_dates if pd.Timestamp(ts) > current]
            if next_dates:
                st.session_state[session_key] = next_dates[0].date()
        except Exception:
            pass

    inspection_date = st.sidebar.date_input('Inspection date', key=session_key)
    return inspection_date.isoformat(), rebalance_frequency


def _mode_service_selector() -> tuple[str, str]:
    usage_mode = st.sidebar.radio('Usage mode', ['Market', 'Standard', 'Tuning', 'Inspection'])
    catalog = {
        'Market': MARKET_SERVICES,
        'Standard': STANDARD_SERVICES,
        'Tuning': TUNING_SERVICES,
        'Inspection': INSPECTION_SERVICES,
    }[usage_mode]
    service_name = st.sidebar.selectbox('Service', list(catalog.keys()))
    st.sidebar.caption(catalog[service_name])
    return usage_mode, service_name


def _estimation_controls(config_defaults: dict[str, Any], *, prefix: str, universe: str) -> tuple[str, int | None]:
    estimation_defaults = config_defaults.get('estimation', {})
    cleaning_default = estimation_defaults.get('cleaning_method', CLEANING_OPTIONS[0])
    config_window_default = int(estimation_defaults.get('covariance_window', 252) or 252)
    cleaning_key = f'{prefix.lower()}::cleaning_method'
    window_key = f'{prefix.lower()}::covariance_window'
    default_key = f'{prefix.lower()}::covariance_window_default'
    context_key = f'{prefix.lower()}::covariance_window_context'

    cleaning = st.sidebar.selectbox(
        f'{prefix} cleaning method',
        CLEANING_OPTIONS,
        index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
        key=cleaning_key,
    )
    st.sidebar.caption(f'Default from config: {cleaning_default}')

    num_assets = len(MARKET_TICKERS.get(universe, []))
    recommended_window = max(2, int(np.ceil(1.5 * num_assets))) if cleaning == 'rie_reference' else config_window_default
    current_context = (universe, cleaning, num_assets)
    previous_context = st.session_state.get(context_key)
    previous_default = st.session_state.get(default_key)

    if previous_context != current_context:
        st.session_state[window_key] = recommended_window
    elif previous_default is None:
        st.session_state[window_key] = recommended_window
    else:
        current_window = int(st.session_state.get(window_key, previous_default))
        if current_window == int(previous_default):
            st.session_state[window_key] = recommended_window

    st.session_state[default_key] = recommended_window
    st.session_state[context_key] = current_context

    covariance_window = int(st.sidebar.number_input(
        f'{prefix} covariance window',
        min_value=2,
        step=1,
        key=window_key,
    ))
    if cleaning == 'rie_reference':
        st.sidebar.caption(f'Default for rie_reference: {recommended_window} (1.5x {num_assets} assets)')
    else:
        st.sidebar.caption(f'Default from config: {config_window_default}')
    return cleaning, covariance_window


def _prepare_table(
    frame: pd.DataFrame,
    *,
    priority: Iterable[str] = (),
    drop: Iterable[str] = (),
    max_rows: int | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    table = frame.copy()
    drop_cols = [column for column in drop if column in table.columns]
    if drop_cols:
        table = table.drop(columns=drop_cols)
    ordered = [column for column in priority if column in table.columns]
    remaining = [column for column in table.columns if column not in ordered]
    table = table.loc[:, ordered + remaining]
    if max_rows is not None:
        table = table.head(max_rows)
    return table


def _render_compact_table(
    frame: pd.DataFrame,
    *,
    priority: Iterable[str] = (),
    drop: Iterable[str] = (),
    max_rows: int | None = None,
    empty_message: str = 'No data available.',
) -> None:
    table = _prepare_table(frame, priority=priority, drop=drop, max_rows=max_rows)
    if table.empty:
        st.info(empty_message)
        return
    column_config = {key: value for key, value in COMMON_COLUMN_CONFIG.items() if key in table.columns}
    height = min(620, max(180, 38 + 35 * min(len(table), 12)))
    st.dataframe(table, use_container_width=True, hide_index=True, column_config=column_config, height=height)


def _render_colored_frame(
    frame: pd.DataFrame,
    *,
    max_rows: int = 120,
    max_cols: int = 16,
    cmap: str = 'RdBu_r',
    empty_message: str = 'No data available.',
) -> None:
    if frame.empty:
        st.info(empty_message)
        return
    preview = frame.iloc[:max_rows, :max_cols].copy()
    styled = preview.style.format('{:.4f}').background_gradient(cmap=cmap, axis=None)
    st.dataframe(styled, use_container_width=True, height=min(720, max(220, 38 + 28 * min(len(preview), 16))))
    if len(frame) > max_rows or frame.shape[1] > max_cols:
        st.caption(f'Showing a preview of {min(len(frame), max_rows)} rows x {min(frame.shape[1], max_cols)} columns. Full data is available in the artifacts.')


def _render_matrix_heatmap(frame: pd.DataFrame, *, title: str, cmap: str = 'RdBu_r') -> None:
    if frame.empty:
        st.info('No matrix available.')
        return
    values = frame.to_numpy(dtype=float)
    fig_size = max(6.0, min(14.0, 0.22 * len(frame) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    vmax = max(vmax, 1e-12)
    image = ax.imshow(values, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    tick_step = max(1, len(frame) // 20)
    positions = np.arange(0, len(frame), tick_step)
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels([str(frame.columns[pos]) for pos in positions], rotation=90, fontsize=7)
    ax.set_yticklabels([str(frame.index[pos]) for pos in positions], fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _render_eigenvalue_distribution_with_mp(
    spectrum_frame: pd.DataFrame,
    *,
    num_assets: int,
    sample_size: int,
) -> None:
    if spectrum_frame.empty:
        st.info('No correlation spectrum available.')
        return

    eigenvalues = pd.to_numeric(spectrum_frame.get('eigenvalue'), errors='coerce').dropna().to_numpy(dtype=float)
    if eigenvalues.size == 0:
        st.info('No correlation eigenvalues available.')
        return

    try:
        mp_law = marchenko_pastur_law(num_assets=num_assets, sample_size=sample_size, variance=1.0)
    except ValueError as exc:
        st.warning(f'Unable to build the Marchenko-Pastur reference law: {exc}')
        return

    grid, density = mp_law.density_grid(num_points=512, padding=0.08)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bin_count = min(36, max(10, int(np.sqrt(eigenvalues.size) * 2.0)))
    hist_values, _, _ = ax.hist(
        eigenvalues,
        bins=bin_count,
        density=True,
        alpha=0.55,
        color='#4C78A8',
        edgecolor='white',
        label='Correlation eigenvalues',
    )
    ax.plot(grid, density, color='#F58518', linewidth=2.0, label='Marchenko-Pastur density')
    ax.axvline(mp_law.lambda_minus, color='#54A24B', linestyle='--', linewidth=1.5, label='MP bulk lower bound')
    ax.axvline(mp_law.lambda_plus, color='#E45756', linestyle='--', linewidth=1.5, label='MP bulk upper bound')

    y_max = float(max(np.max(hist_values) if len(hist_values) else 0.0, np.max(density) if len(density) else 0.0, 1e-6))
    signal_eigenvalues = np.sort(eigenvalues[eigenvalues > mp_law.lambda_plus])
    if signal_eigenvalues.size:
        marker_height = 0.28 * y_max
        ax.vlines(signal_eigenvalues, 0.0, marker_height, colors='#B279A2', linewidth=1.5, alpha=0.8)
        ax.scatter(signal_eigenvalues, np.full_like(signal_eigenvalues, marker_height), color='#B279A2', s=22, zorder=3, label='Signal eigenvalues')

    ax.set_title('Correlation eigenvalue distribution vs Marchenko-Pastur')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.caption(
        'Noise bulk: '
        f'[{mp_law.lambda_minus:.4f}, {mp_law.lambda_plus:.4f}]'
        f' | Signal eigenvalues above lambda+: {signal_eigenvalues.size}/{eigenvalues.size}'
    )


def _render_scree_overlay(
    frame: pd.DataFrame,
    *,
    scenario_column: str,
    title: str,
    log_scale: bool = True,
) -> None:
    if frame.empty:
        st.info('No scree data available.')
        return
    pivot = frame.pivot(index='rank', columns=scenario_column, values='eigenvalue').sort_index()
    if pivot.empty:
        st.info('No scree data available.')
        return

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for column in pivot.columns:
        ax.plot(pivot.index, pivot[column], linewidth=1.8, label=str(column))
    ax.set_title(title)
    ax.set_xlabel('Eigenvalue rank')
    ax.set_ylabel('Eigenvalue')
    if log_scale:
        ax.set_yscale('log')
    ax.grid(alpha=0.2)
    ax.legend(title=scenario_column.replace('_', ' '), fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _artifacts_block(files: dict[str, Path]) -> None:
    if not files:
        st.info('No artifacts available.')
        return
    rows = [{'name': key, 'path': str(value)} for key, value in files.items()]
    _render_compact_table(pd.DataFrame(rows), priority=['name', 'path'])


def _request_block(request: Any, config_defaults: dict[str, Any], resolved: dict[str, Any] | None = None) -> None:
    left, center, right = st.columns(3)
    with left:
        st.caption('Request payload')
        st.json(_json_safe(request), expanded=2)
    with center:
        st.caption('Defaults from config')
        st.json(_json_safe(config_defaults), expanded=2)
    with right:
        st.caption('Resolved context')
        st.json(_json_safe(resolved or {}), expanded=2)


def _service_tabs() -> tuple[Any, Any, Any]:
    return st.tabs(['Results', 'Config', 'Artifacts'])


REFRESH_NEXT_RUN_KEY = 'global::refresh_next_run'


def _queue_force_refresh() -> None:
    st.session_state[REFRESH_NEXT_RUN_KEY] = True


def _consume_refresh_policy() -> str:
    if st.session_state.pop(REFRESH_NEXT_RUN_KEY, False):
        return 'always'
    return 'auto'


def _momentum_display_label(column: str) -> str:
    return MOMENTUM_LABELS.get(column, column)


def _rename_momentum_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: _momentum_display_label(column) for column in frame.columns if column in MOMENTUM_LABELS}
    return frame.rename(columns=rename_map)


def _compact_description(value: object, *, limit: int = 26) -> str:
    text = str(value or '').strip()
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rsplit(' ', 1)[0].strip()
    if not clipped:
        clipped = text[: limit - 1].strip()
    return f"{clipped}..."


def _prepare_market_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    if 'description' in table.columns:
        table['description'] = table['description'].map(_compact_description)
    return table


def _merge_repeated_labels(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    merged = frame.copy()
    if not columns:
        return merged
    previous_values: dict[str, object] = {}
    for idx in merged.index:
        reset_lower = False
        for position, column in enumerate(columns):
            if column not in merged.columns:
                continue
            value = merged.at[idx, column]
            if position > 0 and reset_lower:
                previous_values[column] = None
            if value == previous_values.get(column):
                merged.at[idx, column] = ''
                reset_lower = False
            else:
                previous_values[column] = value
                reset_lower = True
    return merged


def _render_momentum_table(
    frame: pd.DataFrame,
    *,
    empty_message: str = 'No momentum data available.',
    merge_columns: Iterable[str] = (),
) -> None:
    if frame.empty:
        st.info(empty_message)
        return
    table = _prepare_market_display_columns(frame.copy())
    table = _rename_momentum_columns(table)
    table = _merge_repeated_labels(table, [column for column in merge_columns if column in table.columns])
    momentum_cols = [_momentum_display_label(column) for column in MOMENTUM_COLUMNS if _momentum_display_label(column) in table.columns]
    bound = float(np.nanmax(np.abs(table[momentum_cols].to_numpy(dtype=float)))) if momentum_cols else 0.0
    bound = max(bound, 1e-6)
    styled = (
        table.style
        .format({column: '{:.2%}' for column in momentum_cols})
        .background_gradient(cmap='RdYlGn', subset=momentum_cols, vmin=-bound, vmax=bound, axis=None)
    )
    st.table(styled)


def _sort_market_frame(
    frame: pd.DataFrame,
    sort_by: str,
    *,
    sort_mode: str = 'performance',
) -> pd.DataFrame:
    if frame.empty:
        return frame
    if sort_mode == 'hierarchy':
        hierarchy_cols = [column for column in ['sector', 'sub_sector', 'category', 'ticker'] if column in frame.columns]
        if hierarchy_cols:
            return frame.sort_values(hierarchy_cols, kind='stable').reset_index(drop=True)
        return frame.reset_index(drop=True)
    if sort_by not in frame.columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_by, ascending=False, kind='stable').reset_index(drop=True)


def _render_market_overview(
    result: MarketSynthesisResult,
    *,
    ranking_column: str,
    ranking_label: str,
    top_n: int,
    sort_mode: str,
) -> None:
    ticker_detail = result.ticker_frame.reset_index() if isinstance(result.ticker_frame.index, pd.MultiIndex) else result.ticker_frame.reset_index().rename(columns={'index': 'ticker'})
    ranked = _sort_market_frame(ticker_detail, ranking_column, sort_mode=sort_mode)
    positive_share = float((ranked[ranking_column] > 0).mean()) if not ranked.empty else 0.0
    negative_share = float((ranked[ranking_column] < 0).mean()) if not ranked.empty else 0.0
    metric_cols = st.columns(4)
    metric_cols[0].metric('As of date', result.as_of_date.strftime('%Y-%m-%d'))
    metric_cols[1].metric('Universe', result.universe.upper())
    metric_cols[2].metric(f'Positive {ranking_label}', f'{positive_share:.0%}')
    metric_cols[3].metric(f'Negative {ranking_label}', f'{negative_share:.0%}')

    if not ranked.empty:
        leader_cols = st.columns(2)
        best = ranked.iloc[0]
        worst = ranked.iloc[-1]
        best_label = best.get('ticker', best.name)
        worst_label = worst.get('ticker', worst.name)
        if 'sector' in ranked.columns:
            best_label = f"{best_label} ({best.get('sector', '')})".strip()
            worst_label = f"{worst_label} ({worst.get('sector', '')})".strip()
        leader_cols[0].metric(f'Best ticker on {ranking_label}', str(best_label), delta=f"{best[ranking_column]:.2%}")
        leader_cols[1].metric(f'Weakest ticker on {ranking_label}', str(worst_label), delta=f"{worst[ranking_column]:.2%}")

    top_col, bottom_col = st.columns(2)
    with top_col:
        st.caption(f'Top {top_n} tickers on {ranking_label}')
        cols = [column for column in ['sector', 'sub_sector', 'ticker', 'description', *MOMENTUM_COLUMNS] if column in ranked.columns]
        _render_momentum_table(ranked[cols].head(top_n), empty_message='No ticker data available.')
    with bottom_col:
        st.caption(f'Bottom {top_n} tickers on {ranking_label}')
        cols = [column for column in ['sector', 'sub_sector', 'ticker', 'description', *MOMENTUM_COLUMNS] if column in ranked.columns]
        _render_momentum_table(ranked[cols].tail(top_n).sort_values(ranking_column, ascending=True, kind='stable').reset_index(drop=True), empty_message='No ticker data available.')

    if result.has_hierarchy and not result.consolidated_frame.empty:
        sector_frame = _sort_market_frame(result.consolidated_frame.loc[result.consolidated_frame['level'] == 'sector', ['sector', 'num_tickers', *MOMENTUM_COLUMNS]].copy(), ranking_column, sort_mode=sort_mode)
        sub_sector_frame = _sort_market_frame(result.consolidated_frame.loc[result.consolidated_frame['level'] == 'sub_sector', ['sector', 'sub_sector', 'num_tickers', *MOMENTUM_COLUMNS]].copy(), ranking_column, sort_mode=sort_mode)
        summary_cols = st.columns(2)
        with summary_cols[0]:
            st.caption(f'Strongest sectors on {ranking_label}')
            _render_momentum_table(sector_frame.head(min(top_n, 8)), empty_message='No sector summary available.')
        with summary_cols[1]:
            st.caption(f'Strongest sub-sectors on {ranking_label}')
            _render_momentum_table(sub_sector_frame.head(min(top_n, 8)), empty_message='No sub-sector summary available.')


def _render_hyperparameter_results_table(frame: pd.DataFrame) -> None:
    _render_compact_table(
        frame,
        priority=['strategy', 'method', 'covariance_window', 'rebalance_frequency', 'sharpe', 'total_return', 'ann_return', 'ann_vol', 'mdd', 'avg_turnover', 'total_cost', 'final_nav'],
        drop=['covariance_min_periods', 'num_rebalances', 'num_days'],
        empty_message='No hyperparameter results available.',
    )


def _market_nav_chart_frame(frame: pd.DataFrame, *, lookback: str, sampling: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    periods = {'1Y': 252, '2Y': 504}.get(lookback, 252)
    window = frame.tail(min(periods, len(frame))).copy()
    if window.empty:
        return window
    base = window.iloc[0].replace(0.0, np.nan)
    rebased = 100.0 * window.divide(base)
    if sampling == 'Weekly':
        rebased = rebased.resample('W-FRI').last().dropna(how='all')
    return rebased


usage_mode, service_name = _mode_service_selector()
config_path_input = st.sidebar.text_input('Config path', value=DEFAULT_CONFIG)
config_defaults, config_error = _load_defaults(config_path_input)
if config_error:
    st.warning(f'Unable to load config defaults from {config_path_input}: {config_error}')
    config_defaults = {}

universe_default = config_defaults.get('universe', {}).get('name', UNIVERSE_OPTIONS[0])
start_default = config_defaults.get('universe', {}).get('start', '')
universe = _selectbox_with_default('Universe', UNIVERSE_OPTIONS, universe_default)
st.sidebar.caption(f'Default from config: {universe_default}')
start = _date_input_value('Start date', start_default, key='global::start_date')
st.sidebar.caption(f'Default from config: {start_default}')
if st.sidebar.button('Refresh prices now'):
    _queue_force_refresh()
if st.session_state.get(REFRESH_NEXT_RUN_KEY, False):
    st.sidebar.caption('Next run will force-refresh cached prices.')

st.subheader(f'{usage_mode} / {service_name}')

if usage_mode == 'Market' and service_name == 'Momentum synthesis':
    market_date = _latest_or_date_input('Market date', config_defaults.get('evaluation', {}).get('evaluation_end'), key_prefix='market::date', latest_default=True)
    ranking_column = st.sidebar.selectbox('Ranking horizon', MOMENTUM_COLUMNS, format_func=lambda value: MOMENTUM_LABELS[value], index=0)
    sort_mode = st.sidebar.selectbox('Market sort mode', list(MARKET_SORT_OPTIONS), format_func=lambda value: MARKET_SORT_OPTIONS[value], index=0)
    top_n = int(st.sidebar.slider('Top / bottom rows', min_value=3, max_value=12, value=6, step=1))
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/market')
    market_state_key = 'market::result'
    if st.sidebar.button('Run market synthesis'):
        request = MarketSynthesisRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            as_of_date=market_date or None,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        st.session_state[market_state_key] = run_market_synthesis(request)

    result = st.session_state.get(market_state_key)
    if result is not None:
        results_tab, config_tab, artifacts_tab = _service_tabs()
        ranking_label = MOMENTUM_LABELS[ranking_column]
        with results_tab:
            overview_tab, synthesis_tab, details_tab = st.tabs(['Overview', 'Synthesis', 'Details'])
            with overview_tab:
                st.subheader('Momentum synthesis')
                st.caption(f'Cross-sectional snapshot ranked by {ranking_label.lower()} momentum.')
                _render_market_overview(result, ranking_column=ranking_column, ranking_label=ranking_label, top_n=top_n, sort_mode=sort_mode)
            with synthesis_tab:
                synthesis_sort_mode = 'hierarchy' if sort_mode == 'hierarchy' else 'performance'
                if result.synthesis_mode == 'hierarchy' and not result.consolidated_frame.empty:
                    sector_tab, sub_sector_tab = st.tabs(['Sectors', 'Sub-sectors'])
                    sector_frame = _sort_market_frame(
                        result.consolidated_frame.loc[
                            result.consolidated_frame['level'] == 'sector',
                            ['sector', 'num_tickers', *MOMENTUM_COLUMNS],
                        ].copy(),
                        ranking_column,
                        sort_mode=synthesis_sort_mode,
                    )
                    sub_sector_frame = _sort_market_frame(
                        result.consolidated_frame.loc[
                            result.consolidated_frame['level'] == 'sub_sector',
                            ['sector', 'sub_sector', 'num_tickers', *MOMENTUM_COLUMNS],
                        ].copy(),
                        ranking_column,
                        sort_mode=synthesis_sort_mode,
                    )
                    with sector_tab:
                        st.caption('Equal-weight sector portfolios built from the universe constituents.')
                        _render_momentum_table(sector_frame, empty_message='No sector synthesis available.')
                        if not result.sector_nav_frame.empty:
                            nav_left, nav_right = st.columns(2)
                            with nav_left:
                                nav_lookback = st.selectbox('NAV lookback', ['1Y', '2Y'], index=0, key='market::sector_nav_lookback')
                            with nav_right:
                                nav_sampling = st.selectbox('NAV sampling', ['Daily', 'Weekly'], index=0, key='market::sector_nav_sampling')
                            nav_frame = _market_nav_chart_frame(result.sector_nav_frame, lookback=nav_lookback, sampling=nav_sampling)
                            st.caption('Sector equal-weight NAV comparison rebased to 100.')
                            st.line_chart(nav_frame)
                    with sub_sector_tab:
                        st.caption('Equal-weight sub-sector portfolios built from the universe constituents.')
                        _render_momentum_table(sub_sector_frame, empty_message='No sub-sector synthesis available.', merge_columns=['sector'])
                elif result.synthesis_mode == 'category':
                    st.caption('Categories are used here as a reading aid, not as equal-weight portfolio buckets.')
                    detail_frame = result.ticker_frame.reset_index() if isinstance(result.ticker_frame.index, pd.MultiIndex) else result.ticker_frame.reset_index().rename(columns={'index': 'ticker'})
                    detail_frame = _sort_market_frame(detail_frame, ranking_column, sort_mode=sort_mode)
                    categories = detail_frame['category'].drop_duplicates().tolist() if 'category' in detail_frame.columns else []
                    for category in categories:
                        with st.expander(str(category), expanded=False):
                            category_frame = detail_frame.loc[detail_frame['category'] == category, ['ticker', 'description', *MOMENTUM_COLUMNS]].reset_index(drop=True)
                            _render_momentum_table(category_frame, empty_message='No ticker momentum available for this category.')
                else:
                    st.info('No grouping metadata are available for this universe, so only ticker-level momentum is available.')
                    detail_frame = result.ticker_frame.reset_index() if isinstance(result.ticker_frame.index, pd.MultiIndex) else result.ticker_frame.reset_index().rename(columns={'index': 'ticker'})
                    _render_momentum_table(_sort_market_frame(detail_frame, ranking_column, sort_mode=sort_mode), empty_message='No ticker momentum available.')
            with details_tab:
                detail_frame = result.ticker_frame.reset_index() if isinstance(result.ticker_frame.index, pd.MultiIndex) else result.ticker_frame.reset_index().rename(columns={'index': 'ticker'})
                detail_frame = _sort_market_frame(detail_frame, ranking_column, sort_mode=sort_mode)
                if result.synthesis_mode == 'hierarchy' and {'sector', 'sub_sector'}.issubset(detail_frame.columns):
                    sectors = detail_frame['sector'].drop_duplicates().tolist()
                    selected_sector = st.selectbox('Sector', sectors, key='market::selected_sector')
                    sector_frame = detail_frame.loc[detail_frame['sector'] == selected_sector].copy()
                    sub_sector_summary = _sort_market_frame(
                        sector_frame.groupby('sub_sector', as_index=False)[MOMENTUM_COLUMNS].mean(),
                        ranking_column,
                        sort_mode='hierarchy' if sort_mode == 'hierarchy' else 'performance',
                    )
                    counts = sector_frame.groupby('sub_sector').size().rename('num_tickers').reset_index()
                    sub_sector_summary = sub_sector_summary.merge(counts, on='sub_sector', how='left')
                    sub_sector_summary = sub_sector_summary[['sub_sector', 'num_tickers', *MOMENTUM_COLUMNS]]
                    st.caption(f'Sub-sector equal-weight portfolios for {selected_sector}, sorted by {ranking_label.lower()} momentum.')
                    _render_momentum_table(sub_sector_summary, empty_message='No sub-sector detail available for this sector.')
                    st.subheader('Ticker breakdown')
                    for sub_sector in sub_sector_summary['sub_sector'].tolist():
                        with st.expander(str(sub_sector)):
                            ticker_frame = _sort_market_frame(
                                sector_frame.loc[sector_frame['sub_sector'] == sub_sector, ['ticker', 'description', *MOMENTUM_COLUMNS]].reset_index(drop=True),
                                ranking_column,
                                sort_mode=sort_mode,
                            )
                            _render_momentum_table(ticker_frame, empty_message='No ticker detail available for this sub-sector.')
                elif result.synthesis_mode == 'category' and 'category' in detail_frame.columns:
                    categories = detail_frame['category'].drop_duplicates().tolist()
                    selected_category = st.selectbox('Category', categories, key='market::selected_category')
                    category_frame = detail_frame.loc[detail_frame['category'] == selected_category, ['ticker', 'description', *MOMENTUM_COLUMNS]].reset_index(drop=True)
                    st.caption(f'Ticker details for {selected_category}, sorted by {ranking_label.lower()} momentum.')
                    _render_momentum_table(category_frame, empty_message='No ticker detail available for this category.')
                else:
                    st.caption(f'Ticker details sorted by {ranking_label.lower()} momentum.')
                    _render_momentum_table(detail_frame, empty_message='No ticker detail available.')
        with config_tab:
            _request_block(
                result.request,
                config_defaults,
                {
                    'universe': result.universe,
                    'start': result.start,
                    'as_of_date': result.as_of_date,
                    'synthesis_mode': result.synthesis_mode,
                    'has_hierarchy': result.has_hierarchy,
                    'num_tickers': int(len(result.ticker_frame)),
                    'ranking_horizon': ranking_column,
                    'sort_mode': sort_mode,
                    'top_n': top_n,
                },
            )
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Standard' and service_name == 'Allocation':
    allocation_default = config_defaults.get('allocation', {}).get('strategy', STRATEGY_OPTIONS[0])
    strategy = _selectbox_with_default('Strategy', STRATEGY_OPTIONS, allocation_default)
    st.sidebar.caption(f'Default from config: {allocation_default}')
    cleaning_method, covariance_window = _estimation_controls(config_defaults, prefix='Allocation', universe=universe)
    as_of_date = _latest_or_date_input('Allocation date', config_defaults.get('allocation', {}).get('date'), key_prefix='allocation::date', latest_default=config_defaults.get('allocation', {}).get('date') in (None, '', 'None'))
    long_only_default = bool(config_defaults.get('backtest', {}).get('long_only', False))
    long_only = st.sidebar.checkbox('Long only', value=long_only_default)
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/allocation')
    if st.sidebar.button('Run allocation'):
        request = AllocationRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            as_of_date=as_of_date or None,
            strategy=strategy,
            cleaning_method=cleaning_method,
            covariance_window=covariance_window,
            long_only=long_only,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        result = run_allocation(request)
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            st.subheader('Allocation summary')
            _render_compact_table(pd.DataFrame([{
                'universe': result.universe,
                'strategy': result.strategy,
                'cleaning_method': result.cleaning_method,
                'covariance_window': result.covariance_window,
                'allocation_date': str(result.allocation_date.date()),
                'signal_scale': result.signal_scale,
            }]), priority=['universe', 'strategy', 'cleaning_method', 'covariance_window', 'allocation_date', 'signal_scale'])
            st.subheader('Weights')
            _render_compact_table(result.weights.rename('weight').reset_index().rename(columns={'index': 'ticker'}), priority=['ticker', 'weight'])
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'strategy': result.strategy, 'cleaning_method': result.cleaning_method, 'covariance_window': result.covariance_window, 'allocation_date': result.allocation_date, 'signal_scale': result.signal_scale})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Standard' and service_name == 'Evaluation':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    strategy = _selectbox_with_default('Strategy', STRATEGY_OPTIONS, strategy_default)
    st.sidebar.caption(f'Default from config: {strategy_default}')
    cleaning_method, covariance_window = _estimation_controls(config_defaults, prefix='Evaluation', universe=universe)
    freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
    rebalance_frequency = _selectbox_with_default('Rebalance frequency', FREQUENCY_OPTIONS, freq_default)
    st.sidebar.caption(f'Default from config: {freq_default}')
    evaluation_start = _date_input_value('Evaluation start', config_defaults.get('evaluation', {}).get('evaluation_start'), key='standard::evaluation_start')
    evaluation_end = _date_input_value('Evaluation end', config_defaults.get('evaluation', {}).get('evaluation_end'), key='standard::evaluation_end')
    long_only_default = bool(config_defaults.get('backtest', {}).get('long_only', False))
    long_only = st.sidebar.checkbox('Long only', value=long_only_default)
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/evaluation')
    if st.sidebar.button('Run evaluation'):
        request = StandardEvaluationRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            strategy=strategy,
            cleaning_method=cleaning_method,
            covariance_window=covariance_window,
            rebalance_frequency=rebalance_frequency,
            evaluation_start=evaluation_start or None,
            evaluation_end=evaluation_end or None,
            long_only=long_only,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        result = run_evaluation(request)
        nav = (1.0 + result.evaluation_result.daily_returns_net.fillna(0.0)).cumprod()
        benchmark_nav = (1.0 + result.benchmark_returns.fillna(0.0)).cumprod().reindex(nav.index).ffill()
        buy_hold_nav = (1.0 + result.buy_hold_returns.fillna(0.0)).cumprod().reindex(nav.index).ffill()
        nav_frame = pd.DataFrame({
            'optimal_tf portfolio': nav,
            result.benchmark_label: benchmark_nav,
            result.buy_hold_label: buy_hold_nav,
        })
        zero_turnover = pd.Series(0.0, index=result.benchmark_returns.index, dtype=float)
        zero_costs = pd.Series(0.0, index=result.benchmark_returns.index, dtype=float)
        benchmark_summary = evaluation_metrics(result.benchmark_returns, zero_turnover, zero_costs, num_rebalances=0)
        buy_hold_summary = evaluation_metrics(result.buy_hold_returns, zero_turnover, zero_costs, num_rebalances=0)
        summary_rows = [
            {'strategy': result.strategy, **result.evaluation_result.summary.__dict__},
            {'strategy': result.benchmark_label, **benchmark_summary.__dict__},
            {'strategy': result.buy_hold_label, **buy_hold_summary.__dict__},
        ]
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            st.subheader('Summary')
            _render_compact_table(pd.DataFrame(summary_rows), priority=['strategy', 'sharpe', 'total_return', 'ann_return', 'ann_vol', 'mdd', 'avg_turnover', 'total_cost', 'num_rebalances'])
            st.subheader('NAV comparison')
            st.line_chart(nav_frame)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'strategy': result.strategy, 'cleaning_method': result.cleaning_method, 'covariance_window': result.covariance_window, 'rebalance_frequency': result.rebalance_frequency, 'benchmark_label': result.benchmark_label, 'benchmark_metadata': result.benchmark_metadata, 'buy_hold_label': result.buy_hold_label, 'summary': result.evaluation_result.summary.__dict__})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Standard' and service_name == 'Compare':
    compare_defaults = list(config_defaults.get('compare', {}).get('strategies') or [])
    if not compare_defaults:
        compare_defaults = [config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])]
    strategies = _multiselect_with_defaults('Strategies', STRATEGY_OPTIONS, compare_defaults)
    st.sidebar.caption(f"Default from config: {', '.join(compare_defaults)}")
    cleaning_method, covariance_window = _estimation_controls(config_defaults, prefix='Compare', universe=universe)
    freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
    rebalance_frequency = _selectbox_with_default('Rebalance frequency', FREQUENCY_OPTIONS, freq_default)
    st.sidebar.caption(f'Default from config: {freq_default}')
    evaluation_start = _date_input_value('Evaluation start', config_defaults.get('evaluation', {}).get('evaluation_start'), key='compare::evaluation_start')
    evaluation_end = _date_input_value('Evaluation end', config_defaults.get('evaluation', {}).get('evaluation_end'), key='compare::evaluation_end')
    long_only_default = bool(config_defaults.get('backtest', {}).get('long_only', False))
    long_only = st.sidebar.checkbox('Long only', value=long_only_default)
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/compare')
    if st.sidebar.button('Run compare'):
        request = CompareRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            strategies=strategies,
            cleaning_method=cleaning_method,
            covariance_window=covariance_window,
            rebalance_frequency=rebalance_frequency,
            evaluation_start=evaluation_start or None,
            evaluation_end=evaluation_end or None,
            long_only=long_only,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        result = run_compare(request)
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            st.subheader('Summary table')
            _render_compact_table(result.comparison.summary_table, priority=['strategy', 'sharpe', 'total_return', 'ann_return', 'ann_vol', 'mdd', 'avg_turnover', 'total_cost'])
            st.subheader('NAV comparison')
            nav_frame = result.comparison.nav_comparison.copy()
            nav_frame[result.benchmark_label] = result.benchmark_nav.reindex(nav_frame.index).ffill()
            st.line_chart(nav_frame)
            st.subheader('Drawdown comparison')
            drawdown_frame = result.comparison.drawdown_comparison.copy()
            drawdown_frame[result.benchmark_label] = result.benchmark_drawdown.reindex(drawdown_frame.index).ffill()
            st.line_chart(drawdown_frame)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'strategies': result.strategies, 'cleaning_method': result.cleaning_method, 'covariance_window': result.covariance_window, 'rebalance_frequency': result.rebalance_frequency, 'benchmark_label': result.benchmark_label, 'benchmark_metadata': result.benchmark_metadata})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Vary cleaning':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    strategy = _selectbox_with_default('Strategy', STRATEGY_OPTIONS, strategy_default)
    methods_defaults = [config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0]), 'linear_shrinkage']
    methods = _multiselect_with_defaults('Methods', CLEANING_OPTIONS, methods_defaults)
    window_default = int(config_defaults.get('estimation', {}).get('covariance_window', 252) or 252)
    window = st.sidebar.number_input('Window', value=window_default, min_value=2, step=1)
    st.sidebar.caption(f'Default from config: {window_default}')
    log_scale = st.sidebar.checkbox('Scree plot log scale', value=True, key='vary_cleaning::log_scale')
    evaluation_start = _date_input_value('Evaluation start', config_defaults.get('evaluation', {}).get('evaluation_start'), key='vary_cleaning::evaluation_start')
    evaluation_end = _date_input_value('Evaluation end', config_defaults.get('evaluation', {}).get('evaluation_end'), key='vary_cleaning::evaluation_end')
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/vary_cleaning')
    if st.sidebar.button('Run vary cleaning'):
        request = VaryCleaningRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=evaluation_start or None,
            evaluation_end=evaluation_end or None,
            strategy=strategy,
            methods=methods,
            window=int(window),
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
            log_scale=log_scale,
        )
        result = run_vary_cleaning(request)
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            st.subheader('Scenario summary')
            _render_compact_table(result.scenario_summary, priority=['method', 'sharpe', 'total_return', 'ann_return', 'ann_vol', 'mdd', 'avg_turnover', 'total_cost', 'final_nav'])
            st.subheader('Highlights')
            st.json(result.highlights)
            st.subheader('NAV comparison')
            st.line_chart(result.nav_comparison)
            st.subheader('Cleaner scree plot')
            _render_scree_overlay(
                result.scree_frame,
                scenario_column='method',
                title=f'Cleaner scree plot ({strategy}, window={int(window)})',
                log_scale=result.request.log_scale,
            )
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'scenario_key': result.scenario_key, 'covariance_window': int(window), 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Vary window':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    strategy = _selectbox_with_default('Strategy', STRATEGY_OPTIONS, strategy_default)
    cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
    method = _selectbox_with_default('Cleaning method', CLEANING_OPTIONS, cleaning_default)
    windows_default = str(config_defaults.get('estimation', {}).get('covariance_window', 252))
    windows = st.sidebar.text_input('Windows', value=f'{max(20, int(windows_default)//2)},{windows_default},{max(int(windows_default), 252)}')
    log_scale = st.sidebar.checkbox('Scree plot log scale', value=True, key='vary_window::log_scale')
    evaluation_start = _date_input_value('Evaluation start', config_defaults.get('evaluation', {}).get('evaluation_start'), key='vary_window::evaluation_start')
    evaluation_end = _date_input_value('Evaluation end', config_defaults.get('evaluation', {}).get('evaluation_end'), key='vary_window::evaluation_end')
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/vary_window')
    if st.sidebar.button('Run vary window'):
        request = VaryWindowRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=evaluation_start or None,
            evaluation_end=evaluation_end or None,
            strategy=strategy,
            method=method,
            windows=[int(item.strip()) for item in windows.split(',') if item.strip()],
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
            log_scale=log_scale,
        )
        result = run_vary_window(request)
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            st.subheader('Scenario summary')
            _render_compact_table(result.scenario_summary, priority=['covariance_window', 'sharpe', 'total_return', 'ann_return', 'ann_vol', 'mdd', 'avg_turnover', 'total_cost', 'final_nav'])
            st.subheader('NAV comparison')
            st.line_chart(result.nav_comparison)
            st.subheader('Window scree plot')
            _render_scree_overlay(
                result.scree_frame,
                scenario_column='covariance_window',
                title=f'Window scree plot ({strategy}, {method})',
                log_scale=result.request.log_scale,
            )
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'scenario_key': result.scenario_key, 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Vary strategy':
    strategy_defaults = ['RP', config_defaults.get('evaluation', {}).get('strategy', 'ARP'), 'NM']
    strategies = _multiselect_with_defaults('Strategies', STRATEGY_OPTIONS, strategy_defaults)
    cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
    method = _selectbox_with_default('Cleaning method', CLEANING_OPTIONS, cleaning_default)
    window_default = int(config_defaults.get('estimation', {}).get('covariance_window', 252) or 252)
    window = st.sidebar.number_input('Window', value=window_default, min_value=2, step=1)
    evaluation_start = _date_input_value('Evaluation start', config_defaults.get('evaluation', {}).get('evaluation_start'), key='vary_strategy::evaluation_start')
    evaluation_end = _date_input_value('Evaluation end', config_defaults.get('evaluation', {}).get('evaluation_end'), key='vary_strategy::evaluation_end')
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/vary_strategy')
    if st.sidebar.button('Run vary strategy'):
        request = VaryStrategyRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=evaluation_start or None,
            evaluation_end=evaluation_end or None,
            strategies=strategies,
            method=method,
            window=int(window),
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        result = run_vary_strategy(request)
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            st.subheader('Scenario summary')
            _render_compact_table(result.scenario_summary, priority=['strategy', 'sharpe', 'total_return', 'ann_return', 'ann_vol', 'mdd', 'avg_turnover', 'total_cost', 'final_nav'])
            st.subheader('NAV comparison')
            st.line_chart(result.nav_comparison)
            st.subheader('Drawdown comparison')
            st.line_chart(result.drawdown_comparison)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'scenario_key': result.scenario_key, 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Vary frequency':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    strategy = _selectbox_with_default('Strategy', STRATEGY_OPTIONS, strategy_default)
    method, window = _estimation_controls(config_defaults, prefix='Vary frequency', universe=universe)
    freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
    frequencies = _multiselect_with_defaults('Rebalance frequencies', FREQUENCY_OPTIONS, FREQUENCY_OPTIONS)
    st.sidebar.caption(f'Default from config: {freq_default}')
    evaluation_start = _date_input_value('Evaluation start', config_defaults.get('evaluation', {}).get('evaluation_start'), key='vary_frequency::evaluation_start')
    evaluation_end = _date_input_value('Evaluation end', config_defaults.get('evaluation', {}).get('evaluation_end'), key='vary_frequency::evaluation_end')
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/vary_frequency')
    if st.sidebar.button('Run vary frequency'):
        request = VaryFrequencyRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=evaluation_start or None,
            evaluation_end=evaluation_end or None,
            strategy=strategy,
            method=method,
            window=int(window),
            frequencies=frequencies,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        result = run_vary_frequency(request)
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            st.subheader('Scenario summary')
            _render_compact_table(result.scenario_summary, priority=['rebalance_frequency', 'sharpe', 'total_return', 'ann_return', 'ann_vol', 'mdd', 'avg_turnover', 'total_cost', 'final_nav'])
            st.subheader('NAV comparison')
            st.line_chart(result.nav_comparison)
            st.subheader('Drawdown comparison')
            st.line_chart(result.drawdown_comparison)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'scenario_key': result.scenario_key, 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Hyperparameter tuning':
    strategy_defaults = STRATEGY_OPTIONS
    strategies = _multiselect_with_defaults('Strategies', STRATEGY_OPTIONS, strategy_defaults)
    methods = _multiselect_with_defaults('Cleaning methods', CLEANING_OPTIONS, CLEANING_OPTIONS)
    window_default = int(config_defaults.get('estimation', {}).get('covariance_window', 252) or 252)
    windows = st.sidebar.text_input('Windows', value=f'40,60,80,120,{window_default},504')
    freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
    frequencies = _multiselect_with_defaults('Rebalance frequencies', FREQUENCY_OPTIONS, FREQUENCY_OPTIONS)
    st.sidebar.caption(f'Default from config: {freq_default}')
    evaluation_start = _date_input_value('Evaluation start', config_defaults.get('evaluation', {}).get('evaluation_start'), key='hyperparameter::evaluation_start')
    evaluation_end = _date_input_value('Evaluation end', config_defaults.get('evaluation', {}).get('evaluation_end'), key='hyperparameter::evaluation_end')
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/hyperparameter_tuning')
    if st.sidebar.button('Run hyperparameter tuning'):
        request = HyperparameterTuningRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=evaluation_start or None,
            evaluation_end=evaluation_end or None,
            frequencies=frequencies,
            strategies=strategies,
            methods=methods,
            windows=[int(item.strip()) for item in windows.split(',') if item.strip()],
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        result = run_hyperparameter_tuning(request)
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            st.subheader('Results table')
            _render_hyperparameter_results_table(result.results_table)
            st.subheader('Skipped configs')
            _render_compact_table(result.skipped_configs, priority=['strategy', 'method', 'covariance_window', 'rebalance_frequency', 'num_assets', 'reason'], empty_message='No skipped configurations.')
            st.subheader('Highlights')
            st.json(result.highlights)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'num_scenarios': int(len(result.results_table)), 'skipped_configs': int(len(result.skipped_configs)), 'visible_columns': list(_prepare_table(result.results_table, priority=['strategy', 'method', 'covariance_window', 'rebalance_frequency']).columns), 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Inspection' and service_name == 'Inspection snapshot':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    strategy = _selectbox_with_default('Strategy', STRATEGY_OPTIONS, strategy_default)
    st.sidebar.caption(f'Default from config: {strategy_default}')
    cleaning_method, covariance_window = _estimation_controls(config_defaults, prefix='Inspection', universe=universe)
    inspection_date, inspection_frequency = _inspection_date_controls(config_defaults, universe=universe, start=start)
    long_only_default = bool(config_defaults.get('backtest', {}).get('long_only', False))
    long_only = st.sidebar.checkbox('Long only', value=long_only_default)
    output_dir = st.sidebar.text_input('Output dir', value='output/optimal_tf/dashboard/inspection_snapshot')
    if st.sidebar.button('Run inspection snapshot'):
        request = InspectionSnapshotRequest(
            refresh_policy=_consume_refresh_policy(),
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            strategy=strategy,
            date=inspection_date,
            cleaning_method=cleaning_method,
            covariance_window=int(covariance_window),
            long_only=long_only,
            output_dir=output_dir or None,
        )
        result = run_inspection_snapshot(request)
        results_tab, config_tab, artifacts_tab = _service_tabs()
        with results_tab:
            overview_tab, matrices_tab, spectrum_tab, eigenvectors_tab, features_tab, allocation_tab = st.tabs([
                'Overview',
                'Matrices',
                'Spectrum',
                'Correlation eigenvectors',
                'Features',
                'Allocation',
            ])
            with overview_tab:
                st.subheader('Snapshot summary')
                _render_compact_table(pd.DataFrame([{
                    'universe': result.universe,
                    'strategy': result.strategy,
                    'cleaning_method': result.cleaning_method,
                    'covariance_window': result.covariance_window,
                    'allocation_date': str(result.allocation_date.date()),
                    'num_assets': result.num_assets,
                    'sample_size': result.sample_size,
                    'signal_scale': result.signal_scale,
                }]), priority=['universe', 'strategy', 'cleaning_method', 'covariance_window', 'allocation_date', 'num_assets', 'sample_size', 'signal_scale'])
            with matrices_tab:
                st.subheader('Sample correlation heatmap')
                _render_matrix_heatmap(result.sample_correlation, title='Sample correlation')
                st.subheader('Cleaned correlation heatmap')
                _render_matrix_heatmap(result.cleaned_correlation, title='Cleaned correlation')
                st.subheader('Cleaned covariance heatmap')
                _render_matrix_heatmap(result.cleaned_covariance, title='Cleaned covariance', cmap='viridis')
                left, right = st.columns(2)
                with left:
                    st.caption('Sample correlation preview')
                    _render_colored_frame(result.sample_correlation, max_rows=30, max_cols=30)
                with right:
                    st.caption('Cleaned correlation preview')
                    _render_colored_frame(result.cleaned_correlation, max_rows=30, max_cols=30)
            with spectrum_tab:
                st.subheader('Correlation scree plot')
                st.line_chart(result.correlation_spectrum.set_index('rank')['eigenvalue'])
                _render_compact_table(result.correlation_spectrum, priority=['rank', 'eigenvalue', 'variance_share', 'cumulative_variance_share'], max_rows=40)
                st.subheader('Correlation eigenvalue distribution vs Marchenko-Pastur')
                _render_eigenvalue_distribution_with_mp(
                    result.correlation_spectrum,
                    num_assets=result.num_assets,
                    sample_size=result.sample_size,
                )
            with eigenvectors_tab:
                corr_cols = result.correlation_eigenvectors.iloc[:, : min(12, result.correlation_eigenvectors.shape[1])]
                st.subheader('Correlation eigenvectors')
                _render_colored_frame(corr_cols, max_rows=160, max_cols=min(12, corr_cols.shape[1]))
            with features_tab:
                st.subheader('Features at inspection date')
                _render_colored_frame(result.feature_frame, max_rows=200, max_cols=result.feature_frame.shape[1], cmap='RdYlBu_r')
            with allocation_tab:
                st.subheader('Allocation snapshot')
                _render_colored_frame(result.allocation_frame, max_rows=200, max_cols=result.allocation_frame.shape[1], cmap='RdBu_r')
        with config_tab:
            _request_block(result.request, config_defaults, {
                'universe': result.universe,
                'strategy': result.strategy,
                'cleaning_method': result.cleaning_method,
                'covariance_window': result.covariance_window,
                'allocation_date': result.allocation_date,
                'sample_size': result.sample_size,
                'num_assets': result.num_assets,
                'signal_scale': result.signal_scale,
                'inspection_rebalance_frequency': inspection_frequency,
            })
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

else:
    st.info('This inspection UI now focuses on the snapshot workflow only.')
