from __future__ import annotations

from dataclasses import asdict, is_dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from market_tickers_data import MARKET_TICKERS
from optimal_tf.allocation import supported_strategies
from optimal_tf.config_io import load_config
from optimal_tf.data import load_prices_for_universe, load_prices_yf
from optimal_tf.market_fork import build_market_fork_snapshot, write_market_fork_snapshot
from optimal_tf.rebalance import resolve_rebalance_dates, supported_rebalance_frequencies
from optimal_tf.strategies_agnostic import (
    resolve_agnostic_recipe,
    supported_agnostic_strategies,
    supported_normalization_modes,
    supported_q_models,
    supported_signal_models,
)
from optimal_tf.services import (
    AllocationRequest,
    CompareRequest,
    HyperparameterTuningRequest,
    InspectionSnapshotRequest,
    StrategyTestbedRequest,
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
    run_strategy_testbed,
    run_vary_cleaning,
    run_vary_frequency,
    run_vary_strategy,
    run_vary_window,
)
from trading_core.risk import marchenko_pastur_law, supported_cleaning_methods
from trading_core.market.universes import get_universe_benchmark
from trading_core.reporting import (
    cumulative_nav,
    equal_weight_rebalanced_benchmark,
    evaluation_metrics,
    single_asset_buy_and_hold_benchmark,
)
from trading_core.reporting.plots import plt
from trading_core.features import alpha_from_span, effective_span_from_alpha

DEFAULT_CONFIG = 'configs/optimal_tf.example.toml'
UNIVERSE_OPTIONS = sorted(MARKET_TICKERS)
MARKET_UNIVERSES = ['cac40', 'dji', 'eurostoxx50', 'eurostoxx600', 'nasdaq100', 'sbf120', 'sp500']
INDEX_UNIVERSES = ['dataset_all', 'futures', 'index', 'table8_all', 'world_index', 'test']
UNIVERSE_GROUPS = {
    'Markets': [name for name in MARKET_UNIVERSES if name in MARKET_TICKERS],
    'Index universes': [name for name in INDEX_UNIVERSES if name in MARKET_TICKERS],
}
HIDDEN_DASHBOARD_STRATEGIES = {"PHI_0", "PHI_100"}
STRATEGY_OPTIONS = [name for name in supported_strategies() if name not in HIDDEN_DASHBOARD_STRATEGIES]
AGNOSTIC_STRATEGY_OPTIONS = set(supported_agnostic_strategies())
ARP_AGNOSTIC_STRATEGIES = {"ARP_AGNOSTIC", "MARKOWITZ_AGNOSTIC", "PHI_25", "PHI_50"}
ATF_AGNOSTIC_STRATEGIES = {"ATF_AGNOSTIC", "ATF_RAW", "ATF_EMPIRICAL_Q"}
STANDARD_STRATEGY_FAMILIES = ["Classiques", "ARP agnostic", "ATF agnostic"]
STANDARD_STRATEGIES_BY_FAMILY = {
    "Classiques": [name for name in STRATEGY_OPTIONS if name not in ARP_AGNOSTIC_STRATEGIES and name not in ATF_AGNOSTIC_STRATEGIES],
    "ARP agnostic": [name for name in STRATEGY_OPTIONS if name in ARP_AGNOSTIC_STRATEGIES],
    "ATF agnostic": [name for name in STRATEGY_OPTIONS if name in ATF_AGNOSTIC_STRATEGIES],
}
FREQUENCY_OPTIONS = supported_rebalance_frequencies()
CLEANING_OPTIONS = list(supported_cleaning_methods())
AGNOSTIC_SIGNAL_OPTIONS = supported_signal_models()
AGNOSTIC_Q_OPTIONS = supported_q_models()
AGNOSTIC_NORMALIZATION_OPTIONS = supported_normalization_modes()
TESTBED_CUSTOM_PRESET = "CUSTOM_AGNOSTIC"
TESTBED_STRATEGY_FAMILIES = ["Classiques", "ARP agnostic", "ATF agnostic", "Custom agnostic"]
TESTBED_STRATEGIES_BY_FAMILY = {
    "Classiques": ["EW", "RP", "ARP", "NM", "LLTF"],
    "ARP agnostic": ["ARP_AGNOSTIC", "MARKOWITZ_AGNOSTIC", "PHI_25", "PHI_50"],
    "ATF agnostic": ["ATF_AGNOSTIC", "ATF_RAW", "ATF_EMPIRICAL_Q"],
    "Custom agnostic": [TESTBED_CUSTOM_PRESET],
}
MAX_CHART_POINTS = 750

STANDARD_SERVICES = {
    'Allocation': 'Standard allocation of one strategy on one date.',
    'Evaluation': 'Standard backtest using the packaged configuration.',
    'Strategy testbed': 'Custom agnostic backtest with explicit q_model, phi, signal, omega and normalization.',
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

GUIDE_SERVICES = {
    'Strategy guide': 'Short markdown guide describing the currently exposed strategies.',
}

CONFIG_SERVICES = {
    'Config editor': 'Edit the TOML configuration used by optimal_tf directly from the dashboard.',
}

SERVICE_INTRO = {
    ('Guide', 'Strategy guide'): "Reference page for the exposed strategy families and their practical meaning before running a service.",
    ('Config', 'Config editor'): "Administrative view to inspect and update the shared TOML defaults used by the dashboard.",
    ('Standard', 'Allocation'): "Single-date allocation service. Use it when you want the latest portfolio weights rather than a full backtest.",
    ('Standard', 'Evaluation'): "Packaged backtest service. Use it when you want performance, turnover and benchmark comparison over an evaluation window.",
    ('Standard', 'Strategy testbed'): "Research sandbox for one strategy configuration with explicit control over signal, Q, phi, normalization and execution settings.",
    ('Standard', 'Compare'): "Multi-strategy comparison service. It runs several strategies in the same market and backtest context, then compares their outcomes.",
    ('Tuning', 'Vary cleaning'): "Experiment template that keeps the strategy fixed and compares several cleaning methods under the same evaluation setup.",
    ('Tuning', 'Vary window'): "Experiment template that keeps strategy and cleaning fixed while testing several covariance lookback windows.",
    ('Tuning', 'Vary strategy'): "Experiment template that keeps the market context fixed and compares several strategy families under the same estimation settings.",
    ('Tuning', 'Vary frequency'): "Experiment template that keeps the strategy fixed and compares multiple rebalance frequencies as an operational trade-off study.",
    ('Tuning', 'Hyperparameter tuning'): "Advanced search view that evaluates a grid of strategies, cleaning methods, covariance windows and rebalance frequencies.",
    ('Inspection', 'Inspection snapshot'): "Diagnostic snapshot of one dated portfolio state with matrices, spectra, features and allocation views.",
}

STRATEGY_DESCRIPTIONS = {
    "EW": "Equal weight. Baseline long-only allocation equally split across available assets.",
    "RP": "Risk parity proxy. Inverse-volatility baseline built from the covariance diagonal.",
    "ARP": "Agnostic risk parity. Correlation-whitened allocation that spreads risk across eigenmodes.",
    "NM": "Naive Markowitz. Uses the inverse covariance with a flat expected-return vector.",
    "LLTF": "Lead-lag trend following. Empirical cross-asset trend strategy based on virtual return/signal interactions.",
    "ARP_AGNOSTIC": "Eq. 8 agnostic recipe with p = 1 and Q = I. Flat-signal agnostic risk parity.",
    "MARKOWITZ_AGNOSTIC": "Eq. 8 recipe with p = 1 and Q = C. Markowitz-like spectral allocation.",
    "ATF_AGNOSTIC": "Eq. 8 agnostic trend following with p = trend_ema and Q = I.",
    "ATF_RAW": "Same signal as ATF_AGNOSTIC but without gross normalization of the final weights.",
    "ATF_EMPIRICAL_Q": "Agnostic trend following with Q estimated empirically from signal history.",
    "PHI_25": "Eq. 8 phi-family recipe with p = 1 and Q_phi = 0.25 C + 0.75 I.",
    "PHI_50": "Eq. 8 phi-family recipe with p = 1 and Q_phi = 0.5 C + 0.5 I.",
}

COMMON_EVALUATION_DATE_SERVICES = {
    ('Standard', 'Evaluation'),
    ('Standard', 'Strategy testbed'),
    ('Standard', 'Compare'),
    ('Tuning', 'Vary cleaning'),
    ('Tuning', 'Vary window'),
    ('Tuning', 'Vary strategy'),
    ('Tuning', 'Vary frequency'),
    ('Tuning', 'Hyperparameter tuning'),
    ('Inspection', 'Inspection snapshot'),
}

COMMON_COLUMN_CONFIG = {
    'universe': st.column_config.TextColumn('Uni-\nverse', width='small'),
    'strategy': st.column_config.TextColumn('Stra-\ntegy', width='small'),
    'method': st.column_config.TextColumn('Cleaning\nmethod', width='small'),
    'cleaning_method': st.column_config.TextColumn('Cleaning\nmethod', width='small'),
    'covariance_window': st.column_config.NumberColumn('Window\n(days)', width='small', format='%d'),
    'window': st.column_config.NumberColumn('Window\n(days)', width='small', format='%d'),
    'rebalance_frequency': st.column_config.TextColumn('Rebalance\nfrequency', width='small'),
    'sharpe': st.column_config.NumberColumn('Sharpe', width='small', format='%.2f'),
    'ann_return': st.column_config.NumberColumn('Ann\nret\n(%/yr)', width='small', format='%.1f'),
    'cagr': st.column_config.NumberColumn('CAGR\n(%/yr)', width='small', format='%.1f'),
    'ann_vol': st.column_config.NumberColumn('Ann\nvol\n(%/yr)', width='small', format='%.1f'),
    'sortino': st.column_config.NumberColumn('Sortino', width='small', format='%.2f'),
    'skewness': st.column_config.NumberColumn('Skew', width='small', format='%.2f'),
    'mar': st.column_config.NumberColumn('MAR', width='small', format='%.2f'),
    'mdd': st.column_config.NumberColumn('Max\nDD\n(%)', width='small', format='%.1f'),
    'annualized_turnover': st.column_config.NumberColumn('Ann\nturn\n(%/yr)', width='small', format='%.0f'),
    'annualized_cost': st.column_config.NumberColumn('Ann\ntrade cost\n(%/yr)', width='small', format='%.2f'),
    'total_return_gross': st.column_config.NumberColumn('Gross\nret\n(%)', width='small', format='%.1f'),
    'total_return': st.column_config.NumberColumn('Net\nret\n(%)', width='small', format='%.1f'),
    'total_cost': st.column_config.NumberColumn('Trade\ncost pct\n(%)', width='small', format='%.2f'),
    'total_return_cost_drag': st.column_config.NumberColumn('Trade\ncost', width='small', format='%.2f'),
    'avg_turnover': st.column_config.NumberColumn('Average\nturnover\n(%/day)', width='small', format='%.2f'),
    'avg_turnover_per_rebalance': st.column_config.NumberColumn('Average\nturnover\n(%/reb.)', width='small', format='%.1f'),
    'avg_cost_per_rebalance': st.column_config.NumberColumn('Average\ntrade cost\n(%/reb.)', width='small', format='%.2f'),
    'final_nav': st.column_config.NumberColumn('NAV', width='small', format='%.2f'),
    'signal_scale': st.column_config.NumberColumn('Signal\nscale', width='small', format='%.3f'),
    'allocation_date': st.column_config.TextColumn('Allocation\ndate', width='small'),
    'matrix_date': st.column_config.TextColumn('Matrix\ndate', width='small'),
    'rank': st.column_config.NumberColumn('Rank', width='small', format='%d'),
    'eigenvalue': st.column_config.NumberColumn('Eigen\nvalue', width='small', format='%.4f'),
    'variance_share': st.column_config.NumberColumn('Var.\nshare', width='small', format='%.4f'),
    'cumulative_variance_share': st.column_config.NumberColumn('Cum.\nvar', width='small', format='%.4f'),
    'num_assets': st.column_config.NumberColumn('Num\nassets', width='small', format='%d'),
    'sample_size': st.column_config.NumberColumn('Sample\nsize', width='small', format='%d'),
    'num_rebalances': st.column_config.NumberColumn('Number of\nrebalances', width='small', format='%d'),
    'num_days': st.column_config.NumberColumn('Number of\ndays', width='small', format='%d'),
    'reason': st.column_config.TextColumn('Reason', width='medium'),
    'name': st.column_config.TextColumn('Artifact', width='medium'),
    'path': st.column_config.TextColumn('Path', width='large'),
    'ticker': st.column_config.TextColumn('Ticker', width='small'),
    'sector': st.column_config.TextColumn('Sector', width='small'),
    'sub_sector': st.column_config.TextColumn('Sub-\nsector', width='small'),
    'category': st.column_config.TextColumn('Cate-\ngory', width='small'),
    'instrument': st.column_config.TextColumn('Instru-\nment', width='medium'),
    'description': st.column_config.TextColumn('Name', width='medium'),
    'weight': st.column_config.NumberColumn('Weight\n(%)', width='small', format='%.2f'),
    'base_weight': st.column_config.NumberColumn('Base\nweight\n(%)', width='small', format='%.2f'),
    'effective_weight': st.column_config.NumberColumn('Eff.\nweight\n(%)', width='small', format='%.2f'),
    'abs_effective_weight': st.column_config.NumberColumn('Abs eff.\nweight\n(%)', width='small', format='%.2f'),
    'last_price': st.column_config.NumberColumn('Last\nprice', width='small', format='%.4f'),
    'last_return': st.column_config.NumberColumn('Last\nreturn\n(%)', width='small', format='%.2f'),
    'ewma_vol': st.column_config.NumberColumn('EWMA\nvol\n(%)', width='small', format='%.2f'),
    'z_return': st.column_config.NumberColumn('Z\nreturn', width='small', format='%.2f'),
    'trend_signal': st.column_config.NumberColumn('Trend\nsignal', width='small', format='%.2f'),
}

UTILITY_COLUMN_ORDER = [
    'universe', 'sector', 'sub_sector', 'category', 'ticker', 'description', 'instrument',
    'strategy', 'strategies', 'method', 'cleaning_method', 'covariance_window', 'window', 'rebalance_frequency',
    'allocation_date', 'matrix_date', 'num_assets', 'sample_size',
    'signal_scale', 'sharpe', 'ann_return', 'cagr', 'ann_vol', 'sortino', 'skewness', 'mar', 'mdd', 'annualized_turnover', 'annualized_cost',
    'total_return_gross', 'total_return', 'total_cost', 'total_return_cost_drag',
    'final_nav', 'avg_turnover', 'avg_turnover_per_rebalance', 'avg_cost_per_rebalance',
    'num_rebalances', 'num_days',
    'weight', 'base_weight', 'effective_weight', 'abs_effective_weight',
    'benchmark_label', 'buy_hold_label', 'name', 'path', 'reason',
]

st.set_page_config(page_title='optimal_tf dashboard', layout='wide')
st.title('optimal_tf dashboard MVP')
st.caption('Pilotage interactif des services standard, tuning et inspection via l\'API Python.')
st.markdown(
    """
    <style>
    [data-testid="stDataFrame"] div[role="columnheader"],
    [data-testid="stDataFrame"] div[role="gridcell"] {
        font-size: 0.68rem;
        line-height: 0.92;
    }
    [data-testid="stDataFrame"] div[role="columnheader"] *,
    [data-testid="stDataFrame"] div[role="gridcell"] * {
        white-space: pre-line !important;
    }
    [data-testid="stDataFrame"] [data-testid="StyledCell"] {
        padding-top: 0.08rem;
        padding-bottom: 0.08rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        raise ValueError('None values must be handled before TOML serialization.')
    if isinstance(value, (int, float)):
        return repr(value)
    escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def _toml_array(values: Iterable[Any]) -> str:
    return '[' + ', '.join(_toml_scalar(value) for value in values) + ']'


def _build_config_toml(payload: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    section_order = ['universe', 'estimation', 'backtest', 'allocation', 'evaluation', 'compare', 'output']
    for section in section_order:
        values = payload.get(section, {})
        lines.append(f'[{section}]')
        for key, value in values.items():
            if value is None or value == '':
                continue
            if isinstance(value, (list, tuple)):
                lines.append(f'{key} = {_toml_array(value)}')
            else:
                lines.append(f'{key} = {_toml_scalar(value)}')
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def _render_config_editor(config_path: str, config_defaults: dict[str, Any]) -> None:
    st.markdown(
        "Edit the `optimal_tf` TOML config directly from the dashboard. "
        "Saving rewrites the file with the fields exposed below."
    )
    st.caption(f'Editing: `{config_path}`')
    default_save_path = st.session_state.get('config::save_path', config_path)

    universe_defaults = config_defaults.get('universe', {})
    estimation_defaults = config_defaults.get('estimation', {})
    backtest_defaults = config_defaults.get('backtest', {})
    allocation_defaults = config_defaults.get('allocation', {})
    evaluation_defaults = config_defaults.get('evaluation', {})
    compare_defaults = config_defaults.get('compare', {})
    output_defaults = config_defaults.get('output', {})

    with st.form('config_editor_form'):
        st.markdown('### Universe')
        universe_col, start_col = st.columns(2)
        with universe_col:
            universe_name = st.selectbox(
                'Universe',
                UNIVERSE_OPTIONS,
                index=UNIVERSE_OPTIONS.index(universe_defaults.get('name', UNIVERSE_OPTIONS[0])) if universe_defaults.get('name') in UNIVERSE_OPTIONS else 0,
            )
        with start_col:
            universe_start = st.text_input('Start date', value=str(universe_defaults.get('start', '2000-01-01') or '2000-01-01'))

        st.markdown('### Estimation')
        est_left, est_right = st.columns(2)
        with est_left:
            vol_span = int(st.number_input('Vol span', min_value=2, value=int(estimation_defaults.get('vol_span', 60) or 60), step=1))
            covariance_window = int(st.number_input('Covariance window', min_value=2, value=int(estimation_defaults.get('covariance_window', 252) or 252), step=1))
            covariance_min_periods = int(st.number_input('Covariance min periods', min_value=1, value=int(estimation_defaults.get('covariance_min_periods', 252) or 252), step=1))
            max_abs_return = float(st.number_input('Max abs return', min_value=0.0, value=float(estimation_defaults.get('max_abs_return', 1.0) or 1.0), step=0.1))
            cleaning_method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(estimation_defaults.get('cleaning_method', CLEANING_OPTIONS[0])) if estimation_defaults.get('cleaning_method') in CLEANING_OPTIONS else 0,
            )
            linear_shrinkage = float(st.number_input('Linear shrinkage', min_value=0.0, max_value=1.0, value=float(estimation_defaults.get('linear_shrinkage', 0.0) or 0.0), step=0.05))
        with est_right:
            rie_bandwidth = float(st.number_input('RIE bandwidth', min_value=0.0, value=float(estimation_defaults.get('rie_bandwidth', 0.001) or 0.001), step=0.0005, format='%.6f'))
            trend_alpha = float(st.number_input('Trend alpha', min_value=0.0, value=float(estimation_defaults.get('trend_alpha', 0.01575) or 0.01575), step=0.001, format='%.6f'))
            lltf_l2_reg = float(st.number_input('LLTF L2 reg', min_value=0.0, value=float(estimation_defaults.get('lltf_l2_reg', 0.0001) or 0.0001), step=0.0001, format='%.6f'))

        st.markdown('### Backtest')
        bt_left, bt_right = st.columns(2)
        with bt_left:
            sigma_target_annual = float(st.number_input('Sigma target annual', min_value=0.0, value=float(backtest_defaults.get('sigma_target_annual', 0.15) or 0.15), step=0.01))
            portfolio_vol_target = st.checkbox('Portfolio vol target', value=bool(backtest_defaults.get('portfolio_vol_target', True)))
            portfolio_vol_span = int(st.number_input('Portfolio vol span', min_value=2, value=int(backtest_defaults.get('portfolio_vol_span', 60) or 60), step=1))
        with bt_right:
            cost_bps = float(st.number_input('Cost bps', min_value=0.0, value=float(backtest_defaults.get('cost_bps', 0.0) or 0.0), step=1.0))
            weight_smoothing_alpha = float(st.number_input('Weight smoothing alpha', min_value=0.0, max_value=1.0, value=float(backtest_defaults.get('weight_smoothing_alpha', 1.0) or 1.0), step=0.05))
            long_only = st.checkbox('Long only', value=bool(backtest_defaults.get('long_only', False)))

        st.markdown('### Allocation / Evaluation')
        alloc_col, eval_col = st.columns(2)
        with alloc_col:
            allocation_strategy = _strategy_selectbox('Allocation strategy', STRATEGY_OPTIONS, str(allocation_defaults.get('strategy', STRATEGY_OPTIONS[0])), key='config::allocation_strategy')
        with eval_col:
            evaluation_strategy = _strategy_selectbox('Evaluation strategy', STRATEGY_OPTIONS, str(evaluation_defaults.get('strategy', STRATEGY_OPTIONS[0])), key='config::evaluation_strategy')
            rebalance_frequency = st.selectbox(
                'Rebalance frequency',
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(evaluation_defaults.get('rebalance_frequency', FREQUENCY_OPTIONS[0])) if evaluation_defaults.get('rebalance_frequency') in FREQUENCY_OPTIONS else 0,
            )
        eval_start_col, eval_end_col = st.columns(2)
        with eval_start_col:
            evaluation_start = st.text_input('Evaluation start', value=str(evaluation_defaults.get('evaluation_start', '') or ''))
        with eval_end_col:
            evaluation_end = st.text_input('Evaluation end', value=str(evaluation_defaults.get('evaluation_end', '') or ''))

        st.markdown('### Compare')
        compare_strategies = _strategy_selector_columns(
            options=STRATEGY_OPTIONS,
            default_values=list(compare_defaults.get('strategies') or [evaluation_strategy]),
            key='config::compare_strategies',
        )

        st.markdown('### Output')
        out_left, out_right = st.columns(2)
        with out_left:
            allocation_csv = st.text_input('Allocation CSV', value=str(output_defaults.get('allocation_csv', 'output/optimal_tf/weights.csv') or ''))
            allocation_json = st.text_input('Allocation JSON', value=str(output_defaults.get('allocation_json', 'output/optimal_tf/weights.json') or ''))
            evaluation_dir = st.text_input('Evaluation dir', value=str(output_defaults.get('evaluation_dir', 'output/optimal_tf/evaluation_run') or ''))
            evaluation_plot = st.checkbox('Evaluation plot', value=bool(output_defaults.get('evaluation_plot', True)))
        with out_right:
            compare_dir = st.text_input('Compare dir', value=str(output_defaults.get('compare_dir', 'output/optimal_tf/compare_run') or ''))
            compare_clean_dir = st.checkbox('Clean compare dir', value=bool(output_defaults.get('compare_clean_dir', True)))
            compare_plot = st.checkbox('Compare plot', value=bool(output_defaults.get('compare_plot', True)))

        save_col, path_col = st.columns([1, 3])
        with save_col:
            save_clicked = st.form_submit_button('Save config')
        with path_col:
            save_path = st.text_input('Save config as', value=str(default_save_path), help='Destination TOML file path to write when saving.')

    if save_clicked:
        st.session_state['config::save_path'] = save_path
        payload = {
            'universe': {
                'name': universe_name,
                'start': universe_start,
            },
            'estimation': {
                'vol_span': vol_span,
                'covariance_window': covariance_window,
                'covariance_min_periods': covariance_min_periods,
                'max_abs_return': max_abs_return,
                'cleaning_method': cleaning_method,
                'linear_shrinkage': linear_shrinkage,
                'rie_bandwidth': rie_bandwidth,
                'trend_alpha': trend_alpha,
                'lltf_l2_reg': lltf_l2_reg,
            },
            'backtest': {
                'sigma_target_annual': sigma_target_annual,
                'portfolio_vol_target': portfolio_vol_target,
                'portfolio_vol_span': portfolio_vol_span,
                'cost_bps': cost_bps,
                'weight_smoothing_alpha': weight_smoothing_alpha,
                'long_only': long_only,
            },
            'allocation': {
                'strategy': allocation_strategy,
            },
            'evaluation': {
                'strategy': evaluation_strategy,
                'rebalance_frequency': rebalance_frequency,
                'evaluation_start': evaluation_start,
                'evaluation_end': evaluation_end,
            },
            'compare': {
                'strategies': compare_strategies,
            },
            'output': {
                'allocation_csv': allocation_csv,
                'allocation_json': allocation_json,
                'evaluation_dir': evaluation_dir,
                'evaluation_plot': evaluation_plot,
                'compare_dir': compare_dir,
                'compare_clean_dir': compare_clean_dir,
                'compare_plot': compare_plot,
            },
        }
        config_text = _build_config_toml(payload)
        destination = Path(save_path).expanduser()
        if not destination.suffix:
            destination = destination.with_suffix('.toml')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(config_text, encoding='utf-8')
        st.success(f'Config saved to `{destination}`.')
        st.rerun()


def _selectbox_with_default(label: str, options: list[str], default_value: str) -> str:
    default_index = options.index(default_value) if default_value in options else 0
    return st.sidebar.selectbox(label, options, index=default_index)


def _multiselect_with_defaults(label: str, options: list[str], default_values: list[str]) -> list[str]:
    defaults = [item for item in default_values if item in options]
    return st.sidebar.multiselect(label, options, default=defaults)


def _format_universe_label(value: str) -> str:
    labels = {
        'cac40': 'CAC 40',
        'dji': 'DJI',
        'eurostoxx50': 'EURO STOXX 50',
        'eurostoxx600': 'EURO STOXX 600',
        'nasdaq100': 'NASDAQ 100',
        'sbf120': 'SBF 120',
        'sp500': 'S&P 500',
        'dataset_all': 'Dataset all',
        'futures': 'Futures',
        'index': 'Index',
        'table8_all': 'Table 8 all',
        'world_index': 'World index',
        'test': 'Test',
    }
    return labels.get(value, value)


def _default_universe_group(universe: str) -> str:
    for group_name, options in UNIVERSE_GROUPS.items():
        if universe in options:
            return group_name
    return 'Markets'


def _strategy_label(strategy: str) -> str:
    if strategy == "EW":
        return "Baseline | EW"
    if strategy == "RP":
        return "Baseline | RP"
    if strategy in AGNOSTIC_STRATEGY_OPTIONS:
        return f"Agnostic | {strategy}"
    return f"Legacy | {strategy}"


def _strategy_selectbox(label: str, options: list[str], default_value: str, *, key: str | None = None) -> str:
    return _strategy_selector_single_columns(
        label=label,
        options=options,
        default_value=default_value,
        key=key or f"{label.lower().replace(' ', '_')}::single_selector",
    )


def _strategy_multiselect(label: str, options: list[str], default_values: list[str]) -> list[str]:
    st.caption(label)
    return _strategy_selector_columns(
        options=options,
        default_values=default_values,
        key=f"{label.lower().replace(' ', '_')}::multi_selector",
    )


def _strategy_group(strategy: str) -> str:
    if strategy in {"EW", "RP"}:
        return "Baselines"
    if strategy in AGNOSTIC_STRATEGY_OPTIONS:
        return "Agnostic Eq. 8"
    return "Legacy"


def _strategy_selection_error(key: str) -> str | None:
    return st.session_state.get(f"{key}::error")


def _grouped_single_selector(
    *,
    label: str,
    grouped_options: dict[str, list[str]],
    default_value: str,
    key: str,
    format_func: callable | None = None,
    column_widths: list[float] | None = None,
) -> str:
    options = [name for names in grouped_options.values() for name in names]
    default = default_value if default_value in options else options[0]
    selected_key = f"{key}::selected"
    error_key = f"{key}::error"
    if selected_key not in st.session_state or st.session_state[selected_key] not in options:
        st.session_state[selected_key] = default
    selected = st.session_state[selected_key]
    selected_names: list[str] = []
    titles = [title for title, names in grouped_options.items() if names]
    widths = column_widths or [1.0] * max(1, len(titles))
    if len(widths) != len(titles):
        widths = [1.0] * len(titles)
    st.caption(label)
    columns = st.columns(widths)
    formatter = format_func or (lambda value: value)
    for column, title in zip(columns, titles):
        names = grouped_options[title]
        frame = pd.DataFrame(
            {
                "Select": [name == selected for name in names],
                "Strategy": [formatter(name) for name in names],
                "_value": names,
            }
        )
        with column:
            st.caption(title)
            edited = st.data_editor(
                frame,
                key=f"{key}::{title}",
                width="stretch",
                hide_index=True,
                disabled=["Strategy", "_value"],
                column_order=["Select", "Strategy"],
                height=min(460, max(105, 36 + 35 * len(frame))),
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select"),
                    "Strategy": st.column_config.TextColumn("Strategy", width="large"),
                    "_value": None,
                },
            )
            selected_names.extend(edited.loc[edited["Select"], "_value"].tolist())
    if len(selected_names) == 1:
        st.session_state[selected_key] = selected_names[0]
        st.session_state[error_key] = None
    else:
        st.session_state[error_key] = "Select exactly one strategy."
    return str(st.session_state[selected_key])


def _strategy_selector_single_columns(
    *,
    label: str,
    options: list[str],
    default_value: str,
    key: str,
) -> str:
    grouped_options = {
        "Baselines + Legacy": [
            name for name in options if _strategy_group(name) in {"Baselines", "Legacy"}
        ],
        "Agnostic Eq. 8": [
            name for name in options if _strategy_group(name) == "Agnostic Eq. 8"
        ],
    }
    return _grouped_single_selector(
        label=label,
        grouped_options=grouped_options,
        default_value=default_value,
        key=key,
        column_widths=[1.0, 1.5],
    )


def _format_testbed_preset_label(value: str) -> str:
    if value == TESTBED_CUSTOM_PRESET:
        return "Custom agnostic"
    return value


def _single_strategy_family_selector(
    *,
    default_value: str,
    family_key: str,
    strategy_key: str,
) -> str:
    default_family = next(
        (family for family, names in STANDARD_STRATEGIES_BY_FAMILY.items() if default_value in names),
        STANDARD_STRATEGY_FAMILIES[0],
    )
    family_col, strategy_col = st.columns([1.45, 1.0])
    with family_col:
        family = st.radio(
            'Strategy family',
            STANDARD_STRATEGY_FAMILIES,
            index=STANDARD_STRATEGY_FAMILIES.index(default_family),
            key=family_key,
            horizontal=True,
        )
    options = STANDARD_STRATEGIES_BY_FAMILY[family]
    current_strategy = st.session_state.get(strategy_key)
    default_strategy = default_value if default_value in options else options[0]
    if current_strategy not in options:
        st.session_state[strategy_key] = default_strategy
    with strategy_col:
        strategy = st.selectbox(
            'Strategy',
            options,
            index=options.index(st.session_state[strategy_key]),
            key=strategy_key,
        )
    return str(strategy)


def _multi_strategy_family_selector(
    *,
    default_values: list[str],
    family_key: str,
    strategies_key: str,
) -> list[str]:
    default_families = [
        family
        for family, names in STANDARD_STRATEGIES_BY_FAMILY.items()
        if any(value in names for value in default_values)
    ] or [STANDARD_STRATEGY_FAMILIES[0]]
    family_col, strategy_col = st.columns([1.45, 1.0])
    with family_col:
        families = st.multiselect(
            'Strategy families',
            STANDARD_STRATEGY_FAMILIES,
            default=default_families,
            key=family_key,
        )
    active_families = families or default_families
    options = [
        name
        for family in STANDARD_STRATEGY_FAMILIES
        if family in active_families
        for name in STANDARD_STRATEGIES_BY_FAMILY[family]
    ]
    current_strategies = st.session_state.get(strategies_key)
    default_strategies = [value for value in default_values if value in options] or options[: min(3, len(options))]
    if not current_strategies:
        st.session_state[strategies_key] = default_strategies
    else:
        filtered = [value for value in current_strategies if value in options]
        st.session_state[strategies_key] = filtered or default_strategies
    with strategy_col:
        strategies = st.multiselect(
            'Strategies',
            options,
            default=st.session_state[strategies_key],
            key=strategies_key,
        )
    return [str(strategy) for strategy in strategies]


def _testbed_strategy_selector(*, family_key: str, strategy_key: str) -> tuple[str, str]:
    family_col, strategy_col = st.columns([1.45, 1.0])
    with family_col:
        family = st.radio(
            'Strategy family',
            TESTBED_STRATEGY_FAMILIES,
            index=TESTBED_STRATEGY_FAMILIES.index("ARP agnostic"),
            key=family_key,
            horizontal=True,
        )
    options = TESTBED_STRATEGIES_BY_FAMILY[family]
    current_strategy = st.session_state.get(strategy_key)
    default_strategy = options[0] if current_strategy not in options else current_strategy
    if current_strategy not in options:
        st.session_state[strategy_key] = default_strategy
    with strategy_col:
        strategy = st.selectbox(
            'Strategy',
            options,
            index=options.index(default_strategy),
            key=strategy_key,
            format_func=_format_testbed_preset_label,
        )
    return str(family), str(strategy)


def _render_strategy_guide(options: list[str]) -> None:
    groups: dict[str, list[str]] = {"Baselines": [], "Legacy": [], "Agnostic Eq. 8": []}
    for strategy in options:
        groups[_strategy_group(strategy)].append(strategy)

    with st.expander("Strategy guide", expanded=False):
        st.markdown(
            "This guide gives a compact overview of what changes from one strategy to another. "
            "The detailed reference notes still live in the project docs."
        )
        st.markdown(
            """
            ### General principles

            The main difference between the strategies is the way they compute positions.

            What stays mostly shared across strategies:
            - the same historical price inputs,
            - the same rebalance schedule,
            - the same backtest engine for turnover, costs, gross returns, and net returns,
            - the same reporting layer.

            What changes is mainly the position engine:
            - the signal vector,
            - whether the strategy uses no matrix, a covariance matrix, or a correlation matrix,
            - how that object is transformed into portfolio weights.

            ### Eq. 8

            The agnostic research family is organized around:

            `w = omega * C^{-1/2} * Q^{-1/2} * p`

            where `C` is the cleaned correlation matrix, `Q` is the signal correlation matrix or a structural matrix normalized as a correlation matrix, `p` is the signal vector, and `omega` sets the overall scale.

            Intuition:
            - `p` expresses the directional view,
            - `Q` describes how the signal co-moves across assets,
            - `C^{-1/2}` whitens the correlation structure,
            - `omega` rescales the final allocation.

            ### Weight smoothing

            After the strategy computes raw target weights, the portfolio layer can smooth implementation:

            `w_t^{impl} = alpha * w_t^{target} + (1 - alpha) * w_{t-1}^{impl}`

            where `alpha = 1` means no smoothing. Lower `alpha` reduces turnover and costs, but slows the reaction to new estimates and signals.

            ### General workflow

            1. Load prices and define rebalance dates.
            2. Estimate the relevant objects from the historical window.
            3. Compute raw target positions with the selected strategy.
            4. Optionally smooth the weights.
            5. Roll the portfolio over the holding period.
            6. Compute turnover, costs, gross returns, net returns, and summary metrics.

            ### Strategy taxonomy

            **Independent from full covariance/correlation matrices**
            - `EW`: equal-weight baseline with no matrix input.
            - `RP`: inverse-volatility baseline driven by per-asset volatility rather than the full covariance/correlation structure.

            **Using covariance matrices**
            - `NM`: covariance-inverse allocation with a flat expected-return vector.
            - `LLTF`: empirical lead-lag trend strategy using internally estimated EWMA covariance objects on virtual return streams.

            **Using correlation matrices**
            - `ARP`: correlation whitening / mode-balancing baseline.
            - `ARP_AGNOSTIC`, `MARKOWITZ_AGNOSTIC`, `ATF_AGNOSTIC`, `ATF_RAW`, `ATF_EMPIRICAL_Q`, `PHI_25`, `PHI_50`: Eq. 8 research recipes built around cleaned correlation plus different `Q` and `p` choices.
            """
        )
        for title in ("Baselines", "Legacy", "Agnostic Eq. 8"):
            names = groups.get(title, [])
            if not names:
                continue
            lines = [f"**{title}**"]
            for name in names:
                lines.append(f"- `{name}`: {STRATEGY_DESCRIPTIONS.get(name, 'No description available.')}")
            st.markdown("\n".join(lines))


def _strategy_selector_columns(
    *,
    options: list[str],
    default_values: list[str],
    key: str,
) -> list[str]:
    defaults = set(default_values)
    grouped_options = {
        "Baselines + Legacy": [
            name for name in options if _strategy_group(name) in {"Baselines", "Legacy"}
        ],
        "Agnostic Eq. 8": [
            name for name in options if _strategy_group(name) == "Agnostic Eq. 8"
        ],
    }
    selected: set[str] = set()
    columns = st.columns([1.0, 1.5])
    for column, title in zip(columns, ("Baselines + Legacy", "Agnostic Eq. 8")):
        names = grouped_options[title]
        if not names:
            continue
        frame = pd.DataFrame(
            {
                "Select": [name in defaults for name in names],
                "Strategy": names,
            }
        )
        with column:
            st.caption(title)
            edited = st.data_editor(
                frame,
                key=f"{key}::{title}",
                width="stretch",
                hide_index=True,
                disabled=["Strategy"],
                height=min(460, max(140, 36 + 35 * len(frame))),
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select"),
                    "Strategy": st.column_config.TextColumn("Strategy", width="large"),
                },
            )
            selected.update(edited.loc[edited["Select"], "Strategy"].tolist())
    return [name for name in options if name in selected]


def _linear_shrinkage_input(*, key: str, default_value: float) -> float:
    return float(
        st.number_input(
            'Linear shrinkage',
            min_value=0.0,
            max_value=1.0,
            value=float(default_value),
            step=0.05,
            key=key,
        )
    )


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
    usage_mode = st.sidebar.radio(
        'Usage mode',
        ['Guide', 'Config', 'Standard', 'Tuning', 'Inspection'],
        key='nav::usage_mode',
        on_change=_handle_navigation_change,
    )
    catalog = {
        'Guide': GUIDE_SERVICES,
        'Config': CONFIG_SERVICES,
        'Standard': STANDARD_SERVICES,
        'Tuning': TUNING_SERVICES,
        'Inspection': INSPECTION_SERVICES,
    }[usage_mode]
    service_options = list(catalog.keys())
    current_service = st.session_state.get('nav::service_name')
    if current_service not in service_options:
        st.session_state['nav::service_name'] = service_options[0]
    service_name = st.sidebar.selectbox(
        'Service',
        service_options,
        key='nav::service_name',
        on_change=_handle_navigation_change,
    )
    st.sidebar.caption(catalog[service_name])
    return usage_mode, service_name


def _round_up_to_half_ten(value: float) -> int:
    return max(5, int(5 * np.ceil(float(value) / 5.0)))


def _universe_covariance_window_default(universe: str) -> tuple[int, int]:
    num_assets = len(MARKET_TICKERS.get(universe, []))
    return _round_up_to_half_ten(1.5 * max(1, num_assets)), num_assets


def _universe_covariance_window_range_default(universe: str) -> tuple[list[int], int]:
    num_assets = len(MARKET_TICKERS.get(universe, []))
    base = max(1, num_assets)
    windows = sorted({
        _round_up_to_half_ten(multiplier * base)
        for multiplier in (1.1, 1.25, 1.5, 1.75, 2.0)
    })
    if windows[-1] < 252:
        windows.append(252)
    return windows, num_assets


def _sync_widget_default(key: str, default_value: Any, *, context: Any) -> None:
    default_key = f'{key}::default'
    context_key = f'{key}::context'
    previous_context = st.session_state.get(context_key)
    previous_default = st.session_state.get(default_key)

    if previous_context != context:
        st.session_state[key] = default_value
    elif previous_default is None or key not in st.session_state:
        st.session_state[key] = default_value
    else:
        current_value = st.session_state.get(key, previous_default)
        if current_value == previous_default:
            st.session_state[key] = default_value

    st.session_state[default_key] = default_value
    st.session_state[context_key] = context


def _estimation_controls(config_defaults: dict[str, Any], *, prefix: str, universe: str) -> tuple[str, int | None]:
    estimation_defaults = config_defaults.get('estimation', {})
    cleaning_default = estimation_defaults.get('cleaning_method', CLEANING_OPTIONS[0])
    cleaning_key = f'{prefix.lower()}::cleaning_method'
    window_key = f'{prefix.lower()}::covariance_window'

    cleaning = st.sidebar.selectbox(
        f'{prefix} cleaning method',
        CLEANING_OPTIONS,
        index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
        key=cleaning_key,
    )
    st.sidebar.caption(f'Default from config: {cleaning_default}')

    recommended_window, num_assets = _universe_covariance_window_default(universe)
    _sync_widget_default(window_key, recommended_window, context=(universe, cleaning, num_assets))

    covariance_window = int(st.sidebar.number_input(
        f'{prefix} covariance window',
        min_value=2,
        step=1,
        key=window_key,
    ))
    st.sidebar.caption(f'Universe suggestion: {recommended_window} (1.5x {num_assets} assets, rounded up to the nearest 5)')
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
    utility_rank = {column: index for index, column in enumerate(UTILITY_COLUMN_ORDER)}
    remaining = [column for column in table.columns if column not in ordered]
    remaining.sort(key=lambda column: (utility_rank.get(column, 10_000), column))
    table = table.loc[:, ordered + remaining]
    if max_rows is not None:
        table = table.head(max_rows)
    return table


def _compact_metric_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    table = frame.copy()
    percent_columns = {
        'total_cost',
        'ann_return',
        'cagr',
        'ann_vol',
        'annualized_turnover',
        'annualized_cost',
        'mdd',
        'avg_turnover',
        'avg_turnover_per_rebalance',
        'avg_cost_per_rebalance',
        'pct_positive_days',
        'weight',
        'base_weight',
        'effective_weight',
        'abs_effective_weight',
        'last_return',
        'ewma_vol',
    }
    for column in percent_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors='coerce') * 100.0
    return table


def _render_compact_table(
    frame: pd.DataFrame,
    *,
    priority: Iterable[str] = (),
    drop: Iterable[str] = (),
    max_rows: int | None = None,
    empty_message: str = 'No data available.',
) -> None:
    table = _compact_metric_display(_prepare_table(frame, priority=priority, drop=drop, max_rows=max_rows))
    if table.empty:
        st.info(empty_message)
        return
    column_config = {}
    for key, value in COMMON_COLUMN_CONFIG.items():
        if key not in table.columns:
            continue
        if key in {'covariance_window', 'window'} and not pd.api.types.is_numeric_dtype(table[key]):
            column_config[key] = st.column_config.TextColumn('Window\n(days)', width='small')
            continue
        column_config[key] = value
    height = min(560, max(160, 34 + 30 * min(len(table), 12)))
    st.dataframe(table, width="stretch", hide_index=True, column_config=column_config, height=height)


SUMMARY_CORE_METRICS = [
    'sharpe',
    'ann_return',
    'cagr',
    'ann_vol',
    'sortino',
    'skewness',
    'mar',
    'annualized_turnover',
    'annualized_cost',
    'mdd',
    'final_nav',
    'total_return_gross',
    'total_return',
    'total_return_cost_drag',
    'total_cost',
]

SUMMARY_DETAIL_METRICS = [
    'avg_turnover',
    'avg_turnover_per_rebalance',
    'avg_cost_per_rebalance',
    'pct_positive_days',
    'num_rebalances',
]

SUMMARY_COLUMN_LABELS = {
    'strategy': 'Stra<br>tegy',
    'method': 'Cleaning<br>method',
    'covariance_window': 'Window<br>(days)',
    'rebalance_frequency': 'Rebalance<br>freq.',
    'sharpe': 'Sharpe',
    'ann_return': 'Ann<br>ret<br>(%/yr)',
    'cagr': 'CAGR<br>(%/yr)',
    'ann_vol': 'Ann<br>vol<br>(%/yr)',
    'sortino': 'Sortino',
    'skewness': 'Skew',
    'mar': 'MAR',
    'mdd': 'Max<br>DD<br>(%)',
    'annualized_turnover': 'Ann<br>turn<br>(%/yr)',
    'annualized_cost': 'Ann<br>trade cost<br>(%/yr)',
    'total_return_gross': 'Gross<br>return',
    'total_return': 'Net<br>return',
    'total_return_cost_drag': 'Trade<br>cost',
    'total_cost': 'Trade<br>cost pct<br>(%)',
    'final_nav': 'NAV',
    'avg_turnover': 'Avg<br>turn<br>(%/day)',
    'avg_turnover_per_rebalance': 'Avg<br>turn<br>(%/reb.)',
    'avg_cost_per_rebalance': 'Avg<br>trade cost<br>(%/reb.)',
    'pct_positive_days': 'Pos.<br>days<br>(%)',
    'num_days': 'Num<br>days',
    'num_rebalances': 'Num<br>rebal.',
}

SUMMARY_COLUMN_FORMATS = {
    'sharpe': '{:.2f}',
    'ann_return': '{:.1f}',
    'cagr': '{:.1f}',
    'ann_vol': '{:.1f}',
    'sortino': '{:.2f}',
    'skewness': '{:.2f}',
    'mar': '{:.2f}',
    'mdd': '{:.1f}',
    'annualized_turnover': '{:.0f}',
    'annualized_cost': '{:.2f}',
    'total_return_gross': '{:.2f}',
    'total_return': '{:.2f}',
    'total_cost': '{:.2f}',
    'total_return_cost_drag': '{:.2f}',
    'final_nav': '{:.2f}',
    'avg_turnover': '{:.2f}',
    'avg_turnover_per_rebalance': '{:.1f}',
    'avg_cost_per_rebalance': '{:.2f}',
    'pct_positive_days': '{:.1f}',
    'num_days': '{:.0f}',
    'num_rebalances': '{:.0f}',
}


def _load_benchmark_summary_row(
    *,
    config_path: str,
    universe_name: str,
    request_start: str | None,
    target_index: pd.Index,
    scenario_column: str,
    refresh_policy: str = 'auto',
) -> dict[str, Any] | None:
    if len(target_index) == 0:
        return None
    try:
        universe_cfg, estimation, *_ = load_config(config_path)
        effective_start = request_start or universe_cfg.start
        prices = load_prices_for_universe(universe_name, start=effective_start, refresh_policy=refresh_policy)
        benchmark = get_universe_benchmark(universe_name)
        if benchmark and benchmark.get('ticker'):
            benchmark_prices = load_prices_yf([str(benchmark['ticker'])], start=effective_start)
            benchmark_returns = single_asset_buy_and_hold_benchmark(
                benchmark_prices,
                max_abs_return=getattr(estimation, 'max_abs_return', None),
            )
            benchmark_label = str(benchmark.get('name') or benchmark.get('ticker'))
        else:
            benchmark_returns = equal_weight_rebalanced_benchmark(
                prices,
                max_abs_return=getattr(estimation, 'max_abs_return', None),
            )
            benchmark_label = 'universe equal-weight index'
        aligned_returns = benchmark_returns.reindex(pd.Index(target_index)).ffill().fillna(0.0)
        zero_turnover = pd.Series(0.0, index=aligned_returns.index, dtype=float)
        zero_costs = pd.Series(0.0, index=aligned_returns.index, dtype=float)
        benchmark_summary = evaluation_metrics(aligned_returns, zero_turnover, zero_costs, num_rebalances=0)
        benchmark_nav = float(cumulative_nav(aligned_returns).iloc[-1]) if len(aligned_returns) else 1.0
        return {scenario_column: benchmark_label, 'final_nav': benchmark_nav, **benchmark_summary.__dict__}
    except Exception:
        return None


def _append_market_benchmark_row(
    frame: pd.DataFrame,
    *,
    benchmark_label: str | None = None,
    benchmark_summary: dict[str, Any] | None = None,
    config_path: str,
    universe_name: str,
    request_start: str | None,
    target_index: pd.Index,
    scenario_column: str,
    refresh_policy: str = 'auto',
) -> pd.DataFrame:
    if frame.empty or scenario_column not in frame.columns:
        return frame
    if benchmark_label is not None and benchmark_summary is not None:
        benchmark_row = {scenario_column: benchmark_label, **benchmark_summary}
    else:
        benchmark_row = _load_benchmark_summary_row(
            config_path=config_path,
            universe_name=universe_name,
            request_start=request_start,
            target_index=target_index,
            scenario_column=scenario_column,
            refresh_policy=refresh_policy,
        )
    if benchmark_row is None:
        return frame
    benchmark_label = benchmark_row.get(scenario_column)
    existing_labels = frame[scenario_column].astype(str).tolist()
    if benchmark_label in existing_labels:
        return frame
    augmented = pd.concat([frame, pd.DataFrame([benchmark_row])], ignore_index=True)
    if 'sharpe' in augmented.columns:
        augmented = augmented.sort_values('sharpe', ascending=False, na_position='last')
    return augmented.reset_index(drop=True)


def _render_performance_summary_table(
    frame: pd.DataFrame,
    *,
    lead_columns: Iterable[str],
    empty_message: str = 'No data available.',
) -> None:
    if frame.empty:
        st.info(empty_message)
        return
    table = frame.copy()
    lead = [column for column in lead_columns if column in table.columns]
    core = [column for column in SUMMARY_CORE_METRICS if column in table.columns]
    detail = [column for column in SUMMARY_DETAIL_METRICS if column in table.columns]
    shown = [column for column in lead + core + detail if column in table.columns]
    table = _compact_metric_display(table.loc[:, shown])
    for nav_like_column in ('total_return_gross', 'total_return'):
        if nav_like_column in table.columns:
            table[nav_like_column] = pd.to_numeric(table[nav_like_column], errors='coerce') + 1.0
    num_days_value = None
    if 'num_days' in frame.columns:
        numeric_days = pd.to_numeric(frame['num_days'], errors='coerce').dropna()
        if not numeric_days.empty and numeric_days.nunique() == 1:
            num_days_value = int(numeric_days.iloc[0])
        elif not numeric_days.empty:
            num_days_value = f"{int(numeric_days.min())}-{int(numeric_days.max())}"

    def _format_summary_value(column: str, value: Any) -> str:
        if pd.isna(value):
            return ''
        if column in SUMMARY_COLUMN_FORMATS:
            try:
                return SUMMARY_COLUMN_FORMATS[column].format(float(value))
            except Exception:
                return escape(str(value))
        return escape(str(value))

    header_html = ''.join(
        f"<th>{SUMMARY_COLUMN_LABELS.get(column, escape(str(column)).replace('_', '<br>'))}</th>"
        for column in shown
    )
    body_rows = []
    for _, row in table.iterrows():
        cells = ''.join(
            f"<td>{_format_summary_value(column, row[column])}</td>"
            for column in shown
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = ''.join(body_rows)

    st.markdown(
        f"""
        <style>
        .otf-summary-wrap {{
            width: 100%;
            overflow-x: auto;
        }}
        .otf-summary-table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.68rem;
            line-height: 1.0;
        }}
        .otf-summary-table th,
        .otf-summary-table td {{
            border: 1px solid rgba(120, 120, 120, 0.25);
            padding: 0.18rem 0.28rem;
            text-align: center;
            vertical-align: middle;
            word-break: break-word;
        }}
        .otf-summary-table th {{
            background: rgba(120, 120, 120, 0.10);
            font-weight: 600;
        }}
        .otf-summary-table th:first-child,
        .otf-summary-table td:first-child {{
            text-align: left;
            min-width: 8rem;
        }}
        </style>
        <div class="otf-summary-wrap">
          <table class="otf-summary-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{body_html}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if num_days_value is not None:
        st.caption(f'Num days: {num_days_value}')


def _attach_final_nav_from_series(frame: pd.DataFrame, *, label_column: str, nav_by_label: dict[str, float]) -> pd.DataFrame:
    if frame.empty or label_column not in frame.columns:
        return frame
    augmented = frame.copy()
    augmented['final_nav'] = [
        nav_by_label.get(str(label), augmented.iloc[idx]['final_nav'] if 'final_nav' in augmented.columns else np.nan)
        for idx, label in enumerate(augmented[label_column])
    ]
    return augmented


def _display_chart_frame(frame: pd.DataFrame | pd.Series, *, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame | pd.Series:
    if frame.empty or len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    reduced = frame.iloc[::step].copy()
    if reduced.index[-1] != frame.index[-1]:
        reduced = pd.concat([reduced, frame.iloc[[-1]]])
        reduced = reduced[~reduced.index.duplicated(keep="last")]
    return reduced


def _render_line_chart(frame: pd.DataFrame | pd.Series, *, height: int = 280) -> None:
    if isinstance(frame, pd.Series):
        plot = frame.to_frame()
    else:
        plot = frame.copy()
    if plot.empty:
        st.info("No data available for this chart.")
        return
    plot = _display_chart_frame(plot)
    long_frame = plot.reset_index(names="date").melt(id_vars="date", var_name="series", value_name="value")
    chart = (
        alt.Chart(long_frame)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("series:N", legend=alt.Legend(title=None, orient="top")),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".4f"),
            ],
        )
        .properties(height=height)
        .configure(axis=alt.AxisConfig(gridColor="#d7dbe2"))
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, width="stretch")


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
    st.dataframe(styled, width="stretch", height=min(720, max(220, 38 + 28 * min(len(preview), 16))))
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


def _service_result_view(options: list[str], *, key: str) -> str:
    return st.radio(
        'View',
        options,
        key=key,
        horizontal=True,
        label_visibility='collapsed',
    )


def _market_fork_export_block(
    *,
    source_service: str,
    config_path: str,
    universe: str,
    start: str | None,
    as_of_date: str | None,
    source_request: Any,
    source_context: dict[str, Any],
    source_artifacts: dict[str, Path],
    key_prefix: str,
) -> None:
    st.caption('Fork vers market')
    latest_snapshot_key = f'{key_prefix}::latest_snapshot_path'
    output_dir = st.text_input(
        'Fork output dir',
        value='output/optimal_tf/market_forks',
        key=f'{key_prefix}::output_dir',
    )
    if st.button('Create market fork snapshot', key=f'{key_prefix}::create'):
        snapshot = build_market_fork_snapshot(
            source_service=source_service,
            config_path=config_path,
            market_universe=universe,
            market_start=start,
            market_as_of_date=as_of_date,
            source_request=source_request,
            source_context=source_context,
            source_artifacts=source_artifacts,
        )
        snapshot_path = write_market_fork_snapshot(snapshot, output_dir or 'output/optimal_tf/market_forks')
        st.session_state[latest_snapshot_key] = str(snapshot_path)
        st.success(f'Market fork snapshot written to {snapshot_path}')
    latest_snapshot_path = st.session_state.get(latest_snapshot_key)
    if latest_snapshot_path:
        st.caption('Latest snapshot')
        st.code(str(latest_snapshot_path), language='text')
        st.caption(
            "In `market_dashboard`, set `Fork snapshot dir` to this folder and pick the file from "
            "`Recent fork snapshots`."
        )


REFRESH_NEXT_RUN_KEY = 'global::refresh_next_run'


def _queue_force_refresh() -> None:
    st.session_state[REFRESH_NEXT_RUN_KEY] = True


def _consume_refresh_policy() -> str:
    if st.session_state.pop(REFRESH_NEXT_RUN_KEY, False):
        return 'always'
    return 'auto'


def _render_hyperparameter_results_table(frame: pd.DataFrame) -> None:
    trimmed = frame.drop(columns=[column for column in ['covariance_min_periods', 'num_days'] if column in frame.columns])
    _render_performance_summary_table(
        trimmed,
        lead_columns=['strategy', 'method', 'covariance_window', 'rebalance_frequency'],
        empty_message='No hyperparameter results available.',
    )


def _service_signature(mode: str, service: str) -> str:
    return f'{mode}::{service}'


def _handle_navigation_change() -> None:
    st.session_state['nav::pending_signature'] = _service_signature(
        st.session_state.get('nav::usage_mode', 'Guide'),
        st.session_state.get('nav::service_name', 'Strategy guide'),
    )


def _handle_universe_group_change() -> None:
    selected_group = st.session_state.get('nav::universe_group_widget')
    if selected_group is None:
        return
    st.session_state['nav::universe_group'] = selected_group
    group_options = UNIVERSE_GROUPS.get(selected_group, UNIVERSE_OPTIONS) or UNIVERSE_OPTIONS
    current_universe = st.session_state.get('nav::universe')
    if current_universe not in group_options:
        fallback_universe = group_options[0]
        st.session_state['nav::universe'] = fallback_universe
        st.session_state['nav::universe_widget'] = fallback_universe


def _handle_universe_change() -> None:
    selected_universe = st.session_state.get('nav::universe_widget')
    if selected_universe is not None:
        st.session_state['nav::universe'] = selected_universe


def _render_service_header(usage_mode: str, service_name: str) -> None:
    st.subheader(f'{usage_mode} / {service_name}')
    intro = SERVICE_INTRO.get((usage_mode, service_name))
    if intro:
        st.markdown(intro)


def _render_service_guidance(*, defaults: str, action: str, recommendation: str | None = None) -> None:
    with st.container(border=True):
        st.caption(f'Config defaults: {defaults}')
        if recommendation:
            st.caption(f'Recommendation: {recommendation}')
        st.caption(f'This run will: {action}')


usage_mode, service_name = _mode_service_selector()
current_signature = _service_signature(usage_mode, service_name)
pending_signature = st.session_state.pop('nav::pending_signature', None)
rendered_signature = st.session_state.get('nav::rendered_signature')
if pending_signature and pending_signature != current_signature:
    st.session_state['nav::rendered_signature'] = pending_signature
    st.rerun()
if rendered_signature != current_signature:
    st.session_state['nav::rendered_signature'] = current_signature
    if rendered_signature is not None:
        st.rerun()
config_path_input = st.sidebar.text_input('Config path', value=DEFAULT_CONFIG)
config_defaults, config_error = _load_defaults(config_path_input)
if config_error:
    st.warning(f'Unable to load config defaults from {config_path_input}: {config_error}')
    config_defaults = {}

universe_default = config_defaults.get('universe', {}).get('name', UNIVERSE_OPTIONS[0])
start_default = config_defaults.get('universe', {}).get('start', '')
universe = universe_default
start = start_default
common_evaluation_start = config_defaults.get('evaluation', {}).get('evaluation_start')
common_evaluation_end = config_defaults.get('evaluation', {}).get('evaluation_end')
if usage_mode not in {'Guide', 'Config'}:
    group_default = _default_universe_group(universe_default)
    group_names = [name for name, options in UNIVERSE_GROUPS.items() if options]
    stored_group = st.session_state.get('nav::universe_group')
    if stored_group not in group_names:
        stored_group = group_default
        st.session_state['nav::universe_group'] = stored_group
    universe_group = st.sidebar.selectbox(
        'Universe group',
        group_names,
        index=group_names.index(stored_group),
        key='nav::universe_group_widget',
        on_change=_handle_universe_group_change,
    )
    st.session_state['nav::universe_group'] = universe_group
    group_options = UNIVERSE_GROUPS.get(universe_group, UNIVERSE_OPTIONS)
    if not group_options:
        group_options = UNIVERSE_OPTIONS
    stored_universe = st.session_state.get('nav::universe')
    if stored_universe not in group_options:
        stored_universe = universe_default if universe_default in group_options else group_options[0]
        st.session_state['nav::universe'] = stored_universe
    universe = st.sidebar.selectbox(
        'Universe',
        group_options,
        index=group_options.index(stored_universe),
        key='nav::universe_widget',
        on_change=_handle_universe_change,
        format_func=_format_universe_label,
    )
    st.session_state['nav::universe'] = universe
    st.sidebar.caption(f'Default from config: {universe_default}')
    start = _date_input_value('Start date', start_default, key='global::start_date')
    st.sidebar.caption(f'Default from config: {start_default}')
    if (usage_mode, service_name) in COMMON_EVALUATION_DATE_SERVICES:
        common_evaluation_start = _date_input_value(
            'Evaluation start',
            config_defaults.get('evaluation', {}).get('evaluation_start'),
            key='global::evaluation_start',
        )
        common_evaluation_end = _date_input_value(
            'Evaluation end',
            config_defaults.get('evaluation', {}).get('evaluation_end'),
            key='global::evaluation_end',
        )
    if st.sidebar.button('Refresh prices now'):
        _queue_force_refresh()
    if st.session_state.get(REFRESH_NEXT_RUN_KEY, False):
        st.sidebar.caption('Next run will force-refresh cached prices.')

_render_service_header(usage_mode, service_name)

if usage_mode == 'Guide' and service_name == 'Strategy guide':
    st.markdown(
        "This page summarizes the strategy families exposed in the dashboard. "
        "Use it as a quick orientation aid before running allocation, evaluation, or tuning services."
    )
    st.info(
        "Market synthesis now lives in `apps/market_dashboard.py`. "
        "Use `market_dashboard` for market views instead of `optimal_tf_dashboard`."
    )
    _render_strategy_guide(STRATEGY_OPTIONS)
elif usage_mode == 'Config' and service_name == 'Config editor':
    _render_config_editor(config_path_input, config_defaults)
elif usage_mode == 'Standard' and service_name == 'Allocation':
    allocation_default = config_defaults.get('allocation', {}).get('strategy', STRATEGY_OPTIONS[0])
    allocation_state_key = 'standard::allocation::result'
    st.info('Use allocation when you need a dated weight snapshot. Use evaluation instead when you want NAV, turnover and benchmark-relative performance over time.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategy = _single_strategy_family_selector(
            default_value=allocation_default,
            family_key='standard::allocation::strategy_family',
            strategy_key='standard::allocation::strategy',
        )
    st.markdown('### Service parameters')
    with st.form('standard_allocation_form'):
        allocation_date_default = config_defaults.get('allocation', {}).get('date')
        long_only_default = bool(config_defaults.get('backtest', {}).get('long_only', False))
        cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        window_default, num_assets = _universe_covariance_window_default(universe)
        _sync_widget_default('standard::allocation::covariance_window', window_default, context=(universe, 'allocation', num_assets))

        row1 = st.columns(3)
        with row1[0]:
            cleaning_method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
                key='standard::allocation::cleaning_method',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='standard::allocation::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            covariance_window = int(
                st.number_input(
                    'Covariance window',
                    min_value=2,
                    value=int(st.session_state['standard::allocation::covariance_window']),
                    step=1,
                    key='standard::allocation::covariance_window',
                )
            )

        row2 = st.columns(3)
        with row2[0]:
            use_latest_allocation = st.checkbox(
                'Use latest available allocation date',
                value=allocation_date_default in (None, '', 'None'),
                key='standard::allocation::latest_date',
            )
        with row2[1]:
            allocation_date_selected = st.date_input(
                'Allocation date',
                value=_parse_default_date(allocation_date_default).date(),
                key='standard::allocation::date',
                disabled=use_latest_allocation,
            )
            as_of_date = None if use_latest_allocation else pd.Timestamp(allocation_date_selected).date().isoformat()
        with row2[2]:
            long_only = st.checkbox('Long only', value=long_only_default, key='standard::allocation::long_only')

        row3 = st.columns(1)
        with row3[0]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/allocation', key='standard::allocation::output_dir')
        _render_service_guidance(
            defaults=f"strategy={allocation_default}, cleaning={cleaning_default}, covariance window={window_default}",
            recommendation='Prefer the latest available date for an operational snapshot, and switch to a manual date only when you want to inspect a historical allocation.',
            action=f'compute one allocation snapshot for strategy `{strategy}` on `{as_of_date or "the latest available date"}`.',
        )
        st.caption(f'Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)')
        run_clicked = st.form_submit_button('Run allocation')
    if run_clicked:
        request = AllocationRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            as_of_date=as_of_date or None,
            strategy=strategy,
            cleaning_method=cleaning_method,
            linear_shrinkage=linear_shrinkage,
            covariance_window=covariance_window,
            long_only=long_only,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        st.session_state[allocation_state_key] = run_allocation(request)
    result = st.session_state.get(allocation_state_key)
    if result is not None:
        _market_fork_export_block(
            source_service='Allocation',
            config_path=config_path_input,
            universe=result.universe,
            start=result.request.start,
            as_of_date=str(result.allocation_date.date()),
            source_request=result.request,
            source_context={
                'strategy': result.strategy,
                'cleaning_method': result.cleaning_method,
                'covariance_window': result.covariance_window,
                'allocation_date': result.allocation_date,
                'signal_scale': result.signal_scale,
                'top_weights': result.weights.sort_values(ascending=False).head(15),
            },
            source_artifacts=result.artifacts.files,
            key_prefix='standard::allocation::fork',
        )
        summary_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'Config', 'Artifacts'])
        with summary_tab:
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
    eval_state_key = 'standard::evaluation::result'
    st.info('Evaluation is the packaged backtest path: it replays the strategy over an evaluation window and returns NAV, drawdown, turnover, costs and benchmark comparisons.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategy = _single_strategy_family_selector(
            default_value=strategy_default,
            family_key='standard::evaluation::strategy_family',
            strategy_key='standard::evaluation::strategy',
        )
    st.markdown('### Service parameters')
    with st.form('standard_evaluation_form'):
        cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        window_default, num_assets = _universe_covariance_window_default(universe)
        _sync_widget_default('standard::evaluation::covariance_window', window_default, context=(universe, 'evaluation', num_assets))
        freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
        long_only_default = bool(config_defaults.get('backtest', {}).get('long_only', False))
        smoothing_default = float(config_defaults.get('backtest', {}).get('weight_smoothing_alpha', 1.0) or 1.0)

        row1 = st.columns(3)
        with row1[0]:
            cleaning_method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
                key='standard::evaluation::cleaning_method',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='standard::evaluation::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            covariance_window = int(
                st.number_input(
                    'Covariance window',
                    min_value=2,
                    value=int(st.session_state['standard::evaluation::covariance_window']),
                    step=1,
                    key='standard::evaluation::covariance_window',
                )
            )

        row2 = st.columns(3)
        with row2[0]:
            rebalance_frequency = st.selectbox(
                'Rebalance frequency',
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(freq_default) if freq_default in FREQUENCY_OPTIONS else 0,
                key='standard::evaluation::rebalance_frequency',
            )
        with row2[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='standard::evaluation::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )
        with row2[2]:
            long_only = st.checkbox('Long only', value=long_only_default, key='standard::evaluation::long_only')

        row3 = st.columns(1)
        with row3[0]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/evaluation', key='standard::evaluation::output_dir')
        _render_service_guidance(
            defaults=(
                f"strategy={strategy_default}, cleaning={cleaning_default}, covariance window={window_default}, "
                f"rebalance frequency={freq_default}, weight smoothing alpha={smoothing_default}"
            ),
            recommendation='Keep weight smoothing alpha near 1.0 for reference backtests. Lower values mainly help when you want to study turnover and slower reallocation.',
            action=f'run a backtest for strategy `{strategy}` from `{common_evaluation_start or "config start"}` to `{common_evaluation_end or "config end"}`.',
        )
        st.caption(f'Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)')
        run_clicked = st.form_submit_button('Run evaluation')
    if run_clicked:
        request = StandardEvaluationRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            strategy=strategy,
            cleaning_method=cleaning_method,
            linear_shrinkage=linear_shrinkage,
            covariance_window=covariance_window,
            rebalance_frequency=rebalance_frequency,
            weight_smoothing_alpha=weight_smoothing_alpha,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            long_only=long_only,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        st.session_state[eval_state_key] = run_evaluation(request)
    result = st.session_state.get(eval_state_key)
    if result is not None:
        strategy_nav = float(cumulative_nav(result.evaluation_result.daily_returns_net.fillna(0.0)).iloc[-1]) if len(result.evaluation_result.daily_returns_net) else 1.0
        benchmark_nav_value = float(cumulative_nav(result.benchmark_returns.fillna(0.0)).iloc[-1]) if len(result.benchmark_returns) else 1.0
        buy_hold_nav_value = float(cumulative_nav(result.buy_hold_returns.fillna(0.0)).iloc[-1]) if len(result.buy_hold_returns) else 1.0
        zero_turnover = pd.Series(0.0, index=result.benchmark_returns.index, dtype=float)
        zero_costs = pd.Series(0.0, index=result.benchmark_returns.index, dtype=float)
        benchmark_summary = evaluation_metrics(result.benchmark_returns, zero_turnover, zero_costs, num_rebalances=0)
        buy_hold_summary = evaluation_metrics(result.buy_hold_returns, zero_turnover, zero_costs, num_rebalances=0)
        summary_rows = [
            {'strategy': result.strategy, 'final_nav': strategy_nav, **result.evaluation_result.summary.__dict__},
            {'strategy': result.benchmark_label, 'final_nav': benchmark_nav_value, **benchmark_summary.__dict__},
            {'strategy': result.buy_hold_label, 'final_nav': buy_hold_nav_value, **buy_hold_summary.__dict__},
        ]
        _market_fork_export_block(
            source_service='Evaluation',
            config_path=config_path_input,
            universe=result.universe,
            start=result.request.start,
            as_of_date=result.request.evaluation_end or None,
            source_request=result.request,
            source_context={
                'strategy': result.strategy,
                'cleaning_method': result.cleaning_method,
                'covariance_window': result.covariance_window,
                'rebalance_frequency': result.rebalance_frequency,
                'weight_smoothing_alpha': result.request.weight_smoothing_alpha,
                'benchmark_label': result.benchmark_label,
                'buy_hold_label': result.buy_hold_label,
                'summary_rows': summary_rows,
            },
            source_artifacts=result.artifacts.files,
            key_prefix='standard::evaluation::fork',
        )
        summary_tab, nav_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'NAV', 'Config', 'Artifacts'])
        with summary_tab:
            st.subheader('Summary')
            _render_performance_summary_table(pd.DataFrame(summary_rows), lead_columns=['strategy'])
        with nav_tab:
            nav = (1.0 + result.evaluation_result.daily_returns_net.fillna(0.0)).cumprod()
            benchmark_nav = (1.0 + result.benchmark_returns.fillna(0.0)).cumprod().reindex(nav.index).ffill()
            buy_hold_nav = (1.0 + result.buy_hold_returns.fillna(0.0)).cumprod().reindex(nav.index).ffill()
            nav_frame = pd.DataFrame({
                'optimal_tf portfolio': nav,
                result.benchmark_label: benchmark_nav,
                result.buy_hold_label: buy_hold_nav,
            })
            st.subheader('NAV comparison')
            _render_line_chart(nav_frame)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'strategy': result.strategy, 'cleaning_method': result.cleaning_method, 'covariance_window': result.covariance_window, 'rebalance_frequency': result.rebalance_frequency, 'weight_smoothing_alpha': result.request.weight_smoothing_alpha, 'benchmark_label': result.benchmark_label, 'benchmark_metadata': result.benchmark_metadata, 'buy_hold_label': result.buy_hold_label, 'summary': result.evaluation_result.summary.__dict__})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Standard' and service_name == 'Strategy testbed':
    testbed_state_key = 'standard::testbed::result'
    estimation_defaults = config_defaults.get('estimation', {})
    backtest_defaults = config_defaults.get('backtest', {})
    evaluation_defaults = config_defaults.get('evaluation', {})
    testbed_family_key = 'standard::testbed::strategy_family'
    testbed_preset_key = 'standard::testbed::strategy_preset'

    st.markdown('### Strategy')
    with st.container(border=True):
        testbed_family, testbed_preset = _testbed_strategy_selector(
            family_key=testbed_family_key,
            strategy_key=testbed_preset_key,
        )
        if testbed_preset == TESTBED_CUSTOM_PRESET:
            st.caption('`Custom agnostic` unlocks signal, Q, phi, normalization and omega controls.')
        elif testbed_family == 'Classiques':
            st.caption('The selected legacy strategy runs through the shared strategy engine. Agnostic structural controls are disabled because they do not apply.')
        else:
            recipe = resolve_agnostic_recipe(testbed_preset)
            st.caption(
                f"Preset recipe: signal={recipe.signal_model}, q={recipe.q_model}, "
                f"phi={float(recipe.phi):.2f}, norm={recipe.normalization}, omega={float(recipe.omega):.2f}"
            )
            st.caption('The preset locks the structural agnostic controls it defines and keeps the market/backtest controls editable.')

    applied_preset_key = f'{testbed_preset_key}::applied'
    if testbed_preset != st.session_state.get(applied_preset_key):
        if testbed_preset != TESTBED_CUSTOM_PRESET:
            if testbed_family != 'Classiques':
                recipe = resolve_agnostic_recipe(testbed_preset)
                st.session_state['standard::testbed::signal_model'] = recipe.signal_model
                st.session_state['standard::testbed::q_model'] = recipe.q_model
                st.session_state['standard::testbed::phi'] = float(recipe.phi)
                st.session_state['standard::testbed::omega'] = float(recipe.omega)
                st.session_state['standard::testbed::normalization'] = recipe.normalization
        st.session_state[applied_preset_key] = testbed_preset

    agnostic_controls_enabled = testbed_family != 'Classiques'
    preset_locks = {
        'signal_model': (not agnostic_controls_enabled) or testbed_preset != TESTBED_CUSTOM_PRESET,
        'q_model': (not agnostic_controls_enabled) or testbed_preset != TESTBED_CUSTOM_PRESET,
        'phi': (not agnostic_controls_enabled) or testbed_preset != TESTBED_CUSTOM_PRESET,
        'omega': (not agnostic_controls_enabled) or testbed_preset != TESTBED_CUSTOM_PRESET,
        'normalization': (not agnostic_controls_enabled) or testbed_preset != TESTBED_CUSTOM_PRESET,
    }

    st.markdown('### Service parameters')
    with st.form('standard_testbed_form'):
        trend_span_default = int(estimation_defaults.get('trend_span', 252) or 252)
        trend_alpha_default = estimation_defaults.get('trend_alpha', 0.0)
        trend_alpha_value = 0.0 if trend_alpha_default in (None, '') else float(trend_alpha_default)
        smoothing_default = float(backtest_defaults.get('weight_smoothing_alpha', 1.0) or 1.0)
        cleaning_default = estimation_defaults.get('cleaning_method', CLEANING_OPTIONS[0])
        shrinkage_default = float(estimation_defaults.get('linear_shrinkage', 0.0) or 0.0)
        window_default, num_assets = _universe_covariance_window_default(universe)
        _sync_widget_default('standard::testbed::covariance_window', window_default, context=(universe, 'testbed', num_assets))
        freq_default = evaluation_defaults.get('rebalance_frequency', FREQUENCY_OPTIONS[0])

        row1 = st.columns(3)
        with row1[0]:
            signal_model = st.selectbox(
                'Signal model',
                AGNOSTIC_SIGNAL_OPTIONS,
                index=AGNOSTIC_SIGNAL_OPTIONS.index('ones'),
                key='standard::testbed::signal_model',
                disabled=preset_locks['signal_model'],
            )
        trend_disabled = signal_model != 'trend_ema'
        with row1[1]:
            trend_span = int(
                st.number_input(
                    'Trend span',
                    min_value=2,
                    value=trend_span_default,
                    step=1,
                    key='standard::testbed::trend_span',
                    disabled=trend_disabled,
                    help="Used only when Signal model = trend_ema. Ignored otherwise.",
                )
            )
        with row1[2]:
            trend_alpha = float(
                st.number_input(
                    'Trend alpha',
                    min_value=0.0,
                    value=trend_alpha_value,
                    step=0.001,
                    format='%.6f',
                    key='standard::testbed::trend_alpha',
                    disabled=trend_disabled,
                    help="Used only when Signal model = trend_ema. Ignored otherwise.",
                )
            )
        if trend_disabled:
            effective_trend_span = None
            effective_trend_alpha = None
        else:
            base_trend_span = trend_span_default
            base_trend_alpha = trend_alpha_value
            alpha_changed = trend_alpha != base_trend_alpha
            span_changed = trend_span != base_trend_span
            if span_changed and not alpha_changed:
                effective_trend_span = trend_span
                effective_trend_alpha = alpha_from_span(trend_span)
            elif alpha_changed:
                effective_trend_alpha = trend_alpha
                effective_trend_span = effective_span_from_alpha(trend_alpha)
            else:
                effective_trend_span = trend_span
                effective_trend_alpha = trend_alpha

        row2 = st.columns(2)
        with row2[0]:
            q_model = st.selectbox(
                'Q model',
                AGNOSTIC_Q_OPTIONS,
                index=AGNOSTIC_Q_OPTIONS.index('identity'),
                key='standard::testbed::q_model',
                disabled=preset_locks['q_model'],
            )
        with row2[1]:
            phi = float(
                st.number_input(
                    'Phi',
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key='standard::testbed::phi',
                    disabled=preset_locks['phi'],
                    help="Used only when Q model = phi_shrink_correlation. Ignored otherwise.",
                )
            )

        row3 = st.columns(3)
        with row3[0]:
            cleaning_method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
                key='standard::testbed::cleaning_method',
            )
        with row3[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='standard::testbed::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row3[2]:
            covariance_window = int(
                st.number_input(
                    'Covariance window',
                    min_value=2,
                    value=int(st.session_state['standard::testbed::covariance_window']),
                    step=1,
                    key='standard::testbed::covariance_window',
                )
            )

        row4 = st.columns(2)
        with row4[0]:
            normalization = st.selectbox(
                'Normalization',
                AGNOSTIC_NORMALIZATION_OPTIONS,
                index=AGNOSTIC_NORMALIZATION_OPTIONS.index('gross'),
                key='standard::testbed::normalization',
                disabled=preset_locks['normalization'],
            )
        with row4[1]:
            omega = float(
                st.number_input(
                    'Omega',
                    value=1.0,
                    step=0.1,
                    key='standard::testbed::omega',
                    disabled=preset_locks['omega'],
                )
            )

        row5 = st.columns(2)
        with row5[0]:
            rebalance_frequency = st.selectbox(
                'Rebalance frequency',
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(freq_default) if freq_default in FREQUENCY_OPTIONS else 0,
                key='standard::testbed::rebalance_frequency',
            )
        with row5[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='standard::testbed::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )

        row6 = st.columns(2)
        with row6[0]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/testbed', key='standard::testbed::output_dir')
        with row6[1]:
            long_only = st.checkbox(
                'Long only',
                value=bool(backtest_defaults.get('long_only', False)),
                key='standard::testbed::long_only',
            )
        st.caption(
            f"Config defaults: cleaning={cleaning_default}, window={window_default}, "
            f"shrinkage={shrinkage_default}, frequency={freq_default}, smoothing={smoothing_default}, "
            f"long_only={bool(backtest_defaults.get('long_only', False))}"
        )
        if trend_disabled:
            st.caption("`trend span` and `trend alpha` are used only when `Signal model` is `trend_ema`.")
        else:
            st.caption(
                f"Effective trend settings on run: span={effective_trend_span}, "
                f"alpha={float(effective_trend_alpha):.6f}"
            )
        if agnostic_controls_enabled and q_model != 'phi_shrink_correlation':
            st.caption("`phi` is ignored unless `Q model` is `phi_shrink_correlation`.")
        st.caption(f'Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)')
        run_clicked = st.form_submit_button('Run strategy testbed')
    if run_clicked:
        request = StrategyTestbedRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            strategy=None if testbed_preset == TESTBED_CUSTOM_PRESET else testbed_preset,
            cleaning_method=cleaning_method,
            linear_shrinkage=linear_shrinkage,
            covariance_window=covariance_window,
            trend_alpha=None if trend_disabled else trend_alpha,
            trend_span=None if trend_disabled else trend_span,
            rebalance_frequency=rebalance_frequency,
            weight_smoothing_alpha=weight_smoothing_alpha,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            long_only=long_only,
            signal_model=signal_model,
            q_model=q_model,
            phi=phi,
            omega=omega,
            normalization=normalization,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        st.session_state[testbed_state_key] = run_strategy_testbed(request)
    result = st.session_state.get(testbed_state_key)
    if result is not None:
        request_strategy = getattr(result.request, 'strategy', None)
        strategy_nav = float(cumulative_nav(result.evaluation_result.daily_returns_net.fillna(0.0)).iloc[-1]) if len(result.evaluation_result.daily_returns_net) else 1.0
        benchmark_nav_value = float(cumulative_nav(result.benchmark_returns.fillna(0.0)).iloc[-1]) if len(result.benchmark_returns) else 1.0
        buy_hold_nav_value = float(cumulative_nav(result.buy_hold_returns.fillna(0.0)).iloc[-1]) if len(result.buy_hold_returns) else 1.0
        zero_turnover = pd.Series(0.0, index=result.benchmark_returns.index, dtype=float)
        zero_costs = pd.Series(0.0, index=result.benchmark_returns.index, dtype=float)
        benchmark_summary = evaluation_metrics(result.benchmark_returns, zero_turnover, zero_costs, num_rebalances=0)
        buy_hold_summary = evaluation_metrics(result.buy_hold_returns, zero_turnover, zero_costs, num_rebalances=0)
        summary_rows = [
            {'strategy': 'Testbed', 'final_nav': strategy_nav, **result.evaluation_result.summary.__dict__},
            {'strategy': result.benchmark_label, 'final_nav': benchmark_nav_value, **benchmark_summary.__dict__},
            {'strategy': result.buy_hold_label, 'final_nav': buy_hold_nav_value, **buy_hold_summary.__dict__},
        ]
        testbed_parameter_rows = [
            {'parameter': 'strategy', 'value': request_strategy or 'CUSTOM_AGNOSTIC'},
            {'parameter': 'signal_model', 'value': result.signal_model},
            {'parameter': 'trend_span', 'value': result.request.trend_span},
            {'parameter': 'trend_alpha', 'value': result.request.trend_alpha},
            {'parameter': 'q_model', 'value': result.q_model},
            {'parameter': 'phi', 'value': result.phi},
            {'parameter': 'cleaning_method', 'value': result.cleaning_method},
            {'parameter': 'linear_shrinkage', 'value': result.request.linear_shrinkage},
            {'parameter': 'covariance_window', 'value': result.covariance_window},
            {'parameter': 'normalization', 'value': result.normalization},
            {'parameter': 'omega', 'value': result.omega},
            {'parameter': 'rebalance_frequency', 'value': result.rebalance_frequency},
            {'parameter': 'weight_smoothing_alpha', 'value': result.request.weight_smoothing_alpha},
            {'parameter': 'long_only', 'value': result.request.long_only},
        ]
        _market_fork_export_block(
            source_service='Strategy testbed',
            config_path=config_path_input,
            universe=result.universe,
            start=result.request.start,
            as_of_date=result.request.evaluation_end or None,
            source_request=result.request,
            source_context={
                'strategy': request_strategy,
                'strategy_label': result.strategy_label,
                'signal_model': result.signal_model,
                'q_model': result.q_model,
                'phi': result.phi,
                'omega': result.omega,
                'normalization': result.normalization,
                'trend_span': result.request.trend_span,
                'trend_alpha': result.request.trend_alpha,
                'cleaning_method': result.cleaning_method,
                'linear_shrinkage': result.request.linear_shrinkage,
                'covariance_window': result.covariance_window,
                'rebalance_frequency': result.rebalance_frequency,
                'weight_smoothing_alpha': result.request.weight_smoothing_alpha,
                'benchmark_label': result.benchmark_label,
                'buy_hold_label': result.buy_hold_label,
                'summary_rows': summary_rows,
            },
            source_artifacts=result.artifacts.files,
            key_prefix='standard::testbed::fork',
        )
        summary_tab, nav_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'NAV', 'Config', 'Artifacts'])
        with summary_tab:
            st.subheader('Summary')
            _render_performance_summary_table(pd.DataFrame(summary_rows), lead_columns=['strategy'])
            st.caption('Parameters used')
            _render_compact_table(
                pd.DataFrame(testbed_parameter_rows),
                priority=['parameter', 'value'],
            )
        with nav_tab:
            nav = (1.0 + result.evaluation_result.daily_returns_net.fillna(0.0)).cumprod()
            benchmark_nav = (1.0 + result.benchmark_returns.fillna(0.0)).cumprod().reindex(nav.index).ffill()
            buy_hold_nav = (1.0 + result.buy_hold_returns.fillna(0.0)).cumprod().reindex(nav.index).ffill()
            nav_frame = pd.DataFrame({
                'Testbed': nav,
                result.benchmark_label: benchmark_nav,
                result.buy_hold_label: buy_hold_nav,
            })
            st.subheader('NAV comparison')
            _render_line_chart(nav_frame)
        with config_tab:
            _request_block(
                result.request,
                config_defaults,
                {
                    'universe': result.universe,
                    'strategy': request_strategy,
                    'strategy_label': result.strategy_label,
                    'signal_model': result.signal_model,
                    'q_model': result.q_model,
                    'phi': result.phi,
                    'omega': result.omega,
                    'normalization': result.normalization,
                    'trend_span': result.request.trend_span,
                    'trend_alpha': result.request.trend_alpha,
                    'cleaning_method': result.cleaning_method,
                    'linear_shrinkage': result.request.linear_shrinkage,
                    'covariance_window': result.covariance_window,
                    'rebalance_frequency': result.rebalance_frequency,
                    'weight_smoothing_alpha': result.request.weight_smoothing_alpha,
                    'benchmark_label': result.benchmark_label,
                    'benchmark_metadata': result.benchmark_metadata,
                    'buy_hold_label': result.buy_hold_label,
                    'summary': result.evaluation_result.summary.__dict__,
                },
            )
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Standard' and service_name == 'Compare':
    compare_defaults = list(config_defaults.get('compare', {}).get('strategies') or [])
    if not compare_defaults:
        compare_defaults = [config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])]
    compare_state_key = 'standard::compare::result'
    st.info('Compare keeps one market and backtest context fixed, then runs several strategies side by side so you can compare summary metrics, NAV and drawdown.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategies = _multi_strategy_family_selector(
            default_values=compare_defaults,
            family_key='standard::compare::strategy_families',
            strategies_key='standard::compare::strategies',
        )
        st.caption('Select the strategies you want to compare under the same cleaning, window and rebalance assumptions.')
    st.markdown('### Service parameters')
    with st.form('standard_compare_form'):
        cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
        long_only_default = bool(config_defaults.get('backtest', {}).get('long_only', False))
        smoothing_default = float(config_defaults.get('backtest', {}).get('weight_smoothing_alpha', 1.0) or 1.0)
        window_default, num_assets = _universe_covariance_window_default(universe)
        _sync_widget_default('standard::compare::covariance_window', window_default, context=(universe, 'compare', num_assets))
        row1 = st.columns(3)
        with row1[0]:
            cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
            cleaning_method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
                key='standard::compare::cleaning_method',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='standard::compare::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            covariance_window = int(
                st.number_input(
                    'Covariance window',
                    min_value=2,
                    value=int(st.session_state['standard::compare::covariance_window']),
                    step=1,
                    key='standard::compare::covariance_window',
                )
            )

        row2 = st.columns(3)
        with row2[0]:
            rebalance_frequency = st.selectbox(
                'Rebalance frequency',
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(freq_default) if freq_default in FREQUENCY_OPTIONS else 0,
                key='standard::compare::rebalance_frequency',
            )
        with row2[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='standard::compare::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )
        with row2[2]:
            long_only = st.checkbox('Long only', value=long_only_default, key='standard::compare::long_only')

        row3 = st.columns(1)
        with row3[0]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/compare', key='standard::compare::output_dir')
        _render_service_guidance(
            defaults=(
                f"strategies={', '.join(compare_defaults)}, cleaning={cleaning_default}, covariance window={window_default}, "
                f"rebalance frequency={freq_default}, weight smoothing alpha={smoothing_default}"
            ),
            recommendation='Use a compact strategy set when you want a readable NAV comparison. Larger sets are better suited to summary-table inspection first.',
            action=f'compare {max(1, len(strategies))} strategy selections over the shared evaluation window.',
        )
        st.caption(f'Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)')
        run_clicked = st.form_submit_button('Run compare')
    if run_clicked:
        request = CompareRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            strategies=strategies,
            cleaning_method=cleaning_method,
            linear_shrinkage=linear_shrinkage,
            covariance_window=covariance_window,
            rebalance_frequency=rebalance_frequency,
            weight_smoothing_alpha=weight_smoothing_alpha,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            long_only=long_only,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        st.session_state[compare_state_key] = run_compare(request)
    result = st.session_state.get(compare_state_key)
    if result is not None:
        nav_by_label = {
            str(column): float(result.comparison.nav_comparison[column].dropna().iloc[-1])
            for column in result.comparison.nav_comparison.columns
            if result.comparison.nav_comparison[column].dropna().any()
        }
        benchmark_nav_series = result.benchmark_nav.dropna()
        if not benchmark_nav_series.empty:
            nav_by_label[result.benchmark_label] = float(benchmark_nav_series.iloc[-1])
        comparison_summary = _attach_final_nav_from_series(
            result.comparison.summary_table,
            label_column='strategy',
            nav_by_label=nav_by_label,
        )
        _market_fork_export_block(
            source_service='Compare',
            config_path=config_path_input,
            universe=result.universe,
            start=result.request.start,
            as_of_date=result.request.evaluation_end or None,
            source_request=result.request,
            source_context={
                'strategies': result.strategies,
                'cleaning_method': result.cleaning_method,
                'covariance_window': result.covariance_window,
                'rebalance_frequency': result.rebalance_frequency,
                'benchmark_label': result.benchmark_label,
                'summary_table': result.comparison.summary_table,
            },
            source_artifacts=result.artifacts.files,
            key_prefix='standard::compare::fork',
        )
        summary_tab, nav_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'NAV', 'Config', 'Artifacts'])
        with summary_tab:
            st.subheader('Summary table')
            _render_performance_summary_table(comparison_summary, lead_columns=['strategy'])
        with nav_tab:
            st.subheader('NAV comparison')
            nav_frame = result.comparison.nav_comparison.copy()
            nav_frame[result.benchmark_label] = result.benchmark_nav.reindex(nav_frame.index).ffill()
            _render_line_chart(nav_frame)
            st.subheader('Drawdown comparison')
            drawdown_frame = result.comparison.drawdown_comparison.copy()
            drawdown_frame[result.benchmark_label] = result.benchmark_drawdown.reindex(drawdown_frame.index).ffill()
            _render_line_chart(drawdown_frame)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'strategies': result.strategies, 'cleaning_method': result.cleaning_method, 'covariance_window': result.covariance_window, 'rebalance_frequency': result.rebalance_frequency, 'weight_smoothing_alpha': result.request.weight_smoothing_alpha, 'benchmark_label': result.benchmark_label, 'benchmark_metadata': result.benchmark_metadata})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Vary cleaning':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    vary_cleaning_state_key = 'tuning::vary_cleaning::result'
    st.info('This experiment varies the cleaning pipeline only. Strategy, market context and evaluation window stay fixed so the cleaner impact is easier to interpret.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategy = _single_strategy_family_selector(
            default_value=strategy_default,
            family_key='tuning::vary_cleaning::strategy_family',
            strategy_key='tuning::vary_cleaning::strategy',
        )
    st.markdown('### Service parameters')
    with st.form('vary_cleaning_form'):
        methods_defaults = [config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0]), 'linear_shrinkage']
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        window_default, num_assets = _universe_covariance_window_default(universe)
        _sync_widget_default('vary_cleaning::window', window_default, context=(universe, 'vary_cleaning', num_assets))
        smoothing_default = float(config_defaults.get('backtest', {}).get('weight_smoothing_alpha', 1.0) or 1.0)
        freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
        row1 = st.columns(3)
        with row1[0]:
            methods = st.multiselect(
                'Cleaning methods',
                CLEANING_OPTIONS,
                default=[method for method in methods_defaults if method in CLEANING_OPTIONS],
                key='vary_cleaning::methods',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='vary_cleaning::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            window = int(st.number_input('Covariance window', min_value=2, value=int(st.session_state['vary_cleaning::window']), step=1, key='vary_cleaning::window'))

        row2 = st.columns(3)
        with row2[0]:
            rebalance_frequency = st.selectbox(
                'Rebalance frequency',
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(freq_default) if freq_default in FREQUENCY_OPTIONS else 0,
                key='vary_cleaning::rebalance_frequency',
            )
        with row2[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='vary_cleaning::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )
        with row2[2]:
            log_scale = st.checkbox('Scree plot log scale', value=True, key='vary_cleaning::log_scale')

        row3 = st.columns(1)
        with row3[0]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/vary_cleaning', key='vary_cleaning::output_dir')
        _render_service_guidance(
            defaults=f"cleaning methods={', '.join(methods_defaults)}, covariance window={window_default}, rebalance frequency={freq_default}, weight smoothing alpha={smoothing_default}",
            recommendation='Keep the method set short when you want a readable scree overlay. Add more methods only when you are explicitly mapping cleaner sensitivity.',
            action=f'run strategy `{strategy}` across {max(1, len(methods))} cleaning methods.',
        )
        st.caption(f'Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)')
        run_clicked = st.form_submit_button('Run vary cleaning')
    if run_clicked:
        request = VaryCleaningRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            rebalance_frequency=rebalance_frequency,
            strategy=strategy,
            methods=methods,
            linear_shrinkage=linear_shrinkage,
            window=int(window),
            weight_smoothing_alpha=weight_smoothing_alpha,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
            log_scale=log_scale,
        )
        st.session_state[vary_cleaning_state_key] = run_vary_cleaning(request)
    result = st.session_state.get(vary_cleaning_state_key)
    if result is not None:
        summary_tab, nav_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'NAV', 'Config', 'Artifacts'])
        with summary_tab:
            st.subheader('Scenario summary')
            tuning_summary = _append_market_benchmark_row(
                result.strategy_benchmark,
                benchmark_label=result.benchmark_label,
                benchmark_summary=result.benchmark_summary,
                config_path=config_path_input,
                universe_name=result.universe,
                request_start=result.request.start,
                target_index=result.nav_comparison.index,
                scenario_column='method',
            )
            _render_performance_summary_table(tuning_summary, lead_columns=['method'])
            st.subheader('Highlights')
            st.json(result.highlights)
            st.subheader('Cleaner scree plot')
            _render_scree_overlay(
                result.scree_frame,
                scenario_column='method',
                title=f'Cleaner scree plot ({strategy}, window={int(window)})',
                log_scale=result.request.log_scale,
            )
        with nav_tab:
            st.subheader('NAV comparison')
            nav_frame = result.nav_comparison.copy()
            if result.benchmark_label and not result.benchmark_nav.empty:
                nav_frame[result.benchmark_label] = result.benchmark_nav.reindex(nav_frame.index).ffill()
            _render_line_chart(nav_frame)
            st.subheader('Drawdown comparison')
            drawdown_frame = result.drawdown_comparison.copy()
            if result.benchmark_label and not result.benchmark_drawdown.empty:
                drawdown_frame[result.benchmark_label] = result.benchmark_drawdown.reindex(drawdown_frame.index).ffill()
            _render_line_chart(drawdown_frame)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'scenario_key': result.scenario_key, 'covariance_window': int(window), 'rebalance_frequency': result.request.rebalance_frequency, 'weight_smoothing_alpha': result.request.weight_smoothing_alpha, 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Vary window':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    vary_window_state_key = 'tuning::vary_window::result'
    st.info('This experiment varies covariance lookback only. It helps you see how sensitive the strategy is to shorter versus longer estimation windows.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategy = _single_strategy_family_selector(
            default_value=strategy_default,
            family_key='tuning::vary_window::strategy_family',
            strategy_key='tuning::vary_window::strategy',
        )
    st.markdown('### Service parameters')
    with st.form('vary_window_form'):
        cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        window_defaults, num_assets = _universe_covariance_window_range_default(universe)
        window_default = ','.join(str(item) for item in window_defaults)
        _sync_widget_default('vary_window::windows', window_default, context=(universe, 'vary_window', num_assets))
        smoothing_default = float(config_defaults.get('backtest', {}).get('weight_smoothing_alpha', 1.0) or 1.0)
        freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
        row1 = st.columns(3)
        with row1[0]:
            method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
                key='vary_window::method',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='vary_window::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            windows = st.text_input('Covariance windows', value=str(st.session_state['vary_window::windows']), key='vary_window::windows')

        row2 = st.columns(3)
        with row2[0]:
            rebalance_frequency = st.selectbox(
                'Rebalance frequency',
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(freq_default) if freq_default in FREQUENCY_OPTIONS else 0,
                key='vary_window::rebalance_frequency',
            )
        with row2[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='vary_window::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )
        with row2[2]:
            log_scale = st.checkbox('Scree plot log scale', value=True, key='vary_window::log_scale')

        row3 = st.columns(1)
        with row3[0]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/vary_window', key='vary_window::output_dir')
        _render_service_guidance(
            defaults=f"cleaning={cleaning_default}, covariance windows={window_default}, rebalance frequency={freq_default}, weight smoothing alpha={smoothing_default}",
            recommendation='Include one shorter, one default and one longer window to learn something useful without making the comparison noisy.',
            action=f'run strategy `{strategy}` with cleaning `{method}` across the listed covariance windows.',
        )
        st.caption(f'Universe suggestion: {window_default} ({num_assets} assets, rounded up to the nearest 5, with 252 added when needed)')
        run_clicked = st.form_submit_button('Run vary window')
    if run_clicked:
        request = VaryWindowRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            rebalance_frequency=rebalance_frequency,
            strategy=strategy,
            method=method,
            linear_shrinkage=linear_shrinkage,
            windows=[int(item.strip()) for item in windows.split(',') if item.strip()],
            weight_smoothing_alpha=weight_smoothing_alpha,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
            log_scale=log_scale,
        )
        st.session_state[vary_window_state_key] = run_vary_window(request)
    result = st.session_state.get(vary_window_state_key)
    if result is not None:
        summary_tab, nav_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'NAV', 'Config', 'Artifacts'])
        with summary_tab:
            st.subheader('Scenario summary')
            tuning_summary = _append_market_benchmark_row(
                result.strategy_benchmark,
                benchmark_label=result.benchmark_label,
                benchmark_summary=result.benchmark_summary,
                config_path=config_path_input,
                universe_name=result.universe,
                request_start=result.request.start,
                target_index=result.nav_comparison.index,
                scenario_column='covariance_window',
            )
            _render_performance_summary_table(tuning_summary, lead_columns=['covariance_window'])
            st.subheader('Window scree plot')
            _render_scree_overlay(
                result.scree_frame,
                scenario_column='covariance_window',
                title=f'Window scree plot ({strategy}, {method})',
                log_scale=result.request.log_scale,
            )
        with nav_tab:
            st.subheader('NAV comparison')
            nav_frame = result.nav_comparison.copy()
            if result.benchmark_label and not result.benchmark_nav.empty:
                nav_frame[result.benchmark_label] = result.benchmark_nav.reindex(nav_frame.index).ffill()
            _render_line_chart(nav_frame)
            st.subheader('Drawdown comparison')
            drawdown_frame = result.drawdown_comparison.copy()
            if result.benchmark_label and not result.benchmark_drawdown.empty:
                drawdown_frame[result.benchmark_label] = result.benchmark_drawdown.reindex(drawdown_frame.index).ffill()
            _render_line_chart(drawdown_frame)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'scenario_key': result.scenario_key, 'rebalance_frequency': result.request.rebalance_frequency, 'weight_smoothing_alpha': result.request.weight_smoothing_alpha, 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Vary strategy':
    strategy_defaults = ['RP', config_defaults.get('evaluation', {}).get('strategy', 'ARP'), 'NM']
    vary_strategy_state_key = 'tuning::vary_strategy::result'
    st.info('This experiment holds cleaning and covariance window fixed, then varies the strategy choice so you can compare portfolio construction logic directly.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategies = _multi_strategy_family_selector(
            default_values=strategy_defaults,
            family_key='tuning::vary_strategy::strategy_families',
            strategies_key='tuning::vary_strategy::strategies',
        )
        st.caption('Use strategy families to build a focused comparison set rather than one broad kitchen-sink run.')
    st.markdown('### Service parameters')
    with st.form('vary_strategy_form'):
        cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        window_default, num_assets = _universe_covariance_window_default(universe)
        _sync_widget_default('vary_strategy::window', window_default, context=(universe, 'vary_strategy', num_assets))
        smoothing_default = float(config_defaults.get('backtest', {}).get('weight_smoothing_alpha', 1.0) or 1.0)
        freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
        row1 = st.columns(3)
        with row1[0]:
            method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
                key='vary_strategy::method',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='vary_strategy::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            window = int(st.number_input('Covariance window', min_value=2, value=int(st.session_state['vary_strategy::window']), step=1, key='vary_strategy::window'))

        row2 = st.columns(3)
        with row2[0]:
            rebalance_frequency = st.selectbox(
                'Rebalance frequency',
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(freq_default) if freq_default in FREQUENCY_OPTIONS else 0,
                key='vary_strategy::rebalance_frequency',
            )
        with row2[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='vary_strategy::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )
        with row2[2]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/vary_strategy', key='vary_strategy::output_dir')
        _render_service_guidance(
            defaults=f"cleaning={cleaning_default}, covariance window={window_default}, rebalance frequency={freq_default}, weight smoothing alpha={smoothing_default}",
            recommendation='Mix only a few families per run if you want the NAV chart to stay readable and the result table easier to interpret.',
            action=f'compare {max(1, len(strategies))} strategies under one shared cleaning and covariance window setup.',
        )
        st.caption(f'Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)')
        run_clicked = st.form_submit_button('Run vary strategy')
    if run_clicked:
        request = VaryStrategyRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            rebalance_frequency=rebalance_frequency,
            strategies=strategies,
            method=method,
            linear_shrinkage=linear_shrinkage,
            window=int(window),
            weight_smoothing_alpha=weight_smoothing_alpha,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        st.session_state[vary_strategy_state_key] = run_vary_strategy(request)
    result = st.session_state.get(vary_strategy_state_key)
    if result is not None:
        summary_tab, nav_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'NAV', 'Config', 'Artifacts'])
        with summary_tab:
            st.subheader('Scenario summary')
            tuning_summary = _append_market_benchmark_row(
                result.strategy_benchmark,
                benchmark_label=result.benchmark_label,
                benchmark_summary=result.benchmark_summary,
                config_path=config_path_input,
                universe_name=result.universe,
                request_start=result.request.start,
                target_index=result.nav_comparison.index,
                scenario_column='strategy',
            )
            _render_performance_summary_table(tuning_summary, lead_columns=['strategy'])
        with nav_tab:
            st.subheader('NAV comparison')
            nav_frame = result.nav_comparison.copy()
            if result.benchmark_label and not result.benchmark_nav.empty:
                nav_frame[result.benchmark_label] = result.benchmark_nav.reindex(nav_frame.index).ffill()
            _render_line_chart(nav_frame)
            st.subheader('Drawdown comparison')
            drawdown_frame = result.drawdown_comparison.copy()
            if result.benchmark_label and not result.benchmark_drawdown.empty:
                drawdown_frame[result.benchmark_label] = result.benchmark_drawdown.reindex(drawdown_frame.index).ffill()
            _render_line_chart(drawdown_frame)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'scenario_key': result.scenario_key, 'rebalance_frequency': result.request.rebalance_frequency, 'weight_smoothing_alpha': result.request.weight_smoothing_alpha, 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Vary frequency':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    vary_frequency_state_key = 'tuning::vary_frequency::result'
    st.info('This experiment varies rebalance cadence only. It helps you study the trade-off between faster reaction, turnover and implementation friction.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategy = _single_strategy_family_selector(
            default_value=strategy_default,
            family_key='tuning::vary_frequency::strategy_family',
            strategy_key='tuning::vary_frequency::strategy',
        )
    st.markdown('### Service parameters')
    with st.form('vary_frequency_form'):
        cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
        smoothing_default = float(config_defaults.get('backtest', {}).get('weight_smoothing_alpha', 1.0) or 1.0)
        window_default, num_assets = _universe_covariance_window_default(universe)
        _sync_widget_default('vary_frequency::window', window_default, context=(universe, 'vary_frequency', num_assets))
        row1 = st.columns(3)
        with row1[0]:
            method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
                key='vary_frequency::method',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='vary_frequency::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            window = int(
                st.number_input(
                    'Covariance window',
                    min_value=2,
                    value=int(st.session_state['vary_frequency::window']),
                    step=1,
                    key='vary_frequency::window',
                )
            )

        row2 = st.columns(2)
        with row2[0]:
            frequencies = st.multiselect(
                'Rebalance frequencies',
                FREQUENCY_OPTIONS,
                default=FREQUENCY_OPTIONS,
                key='vary_frequency::frequencies',
            )
        with row2[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='vary_frequency::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )

        row3 = st.columns(1)
        with row3[0]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/vary_frequency', key='vary_frequency::output_dir')
        _render_service_guidance(
            defaults=f"cleaning={cleaning_default}, rebalance frequency={freq_default}, covariance window={window_default}, weight smoothing alpha={smoothing_default}",
            recommendation='Group frequencies as a comparison set, then use the NAV and drawdown tabs to see whether higher activity is really rewarded.',
            action=f'run strategy `{strategy}` across {max(1, len(frequencies))} rebalance frequencies.',
        )
        st.caption(f'Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)')
        run_clicked = st.form_submit_button('Run vary frequency')
    if run_clicked:
        request = VaryFrequencyRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            strategy=strategy,
            method=method,
            linear_shrinkage=linear_shrinkage,
            window=int(window),
            frequencies=frequencies,
            weight_smoothing_alpha=weight_smoothing_alpha,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        st.session_state[vary_frequency_state_key] = run_vary_frequency(request)
    result = st.session_state.get(vary_frequency_state_key)
    if result is not None:
        summary_tab, nav_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'NAV', 'Config', 'Artifacts'])
        with summary_tab:
            st.subheader('Scenario summary')
            tuning_summary = _append_market_benchmark_row(
                result.strategy_benchmark,
                benchmark_label=result.benchmark_label,
                benchmark_summary=result.benchmark_summary,
                config_path=config_path_input,
                universe_name=result.universe,
                request_start=result.request.start,
                target_index=result.nav_comparison.index,
                scenario_column='rebalance_frequency',
            )
            _render_performance_summary_table(tuning_summary, lead_columns=['rebalance_frequency'])
        with nav_tab:
            st.subheader('NAV comparison')
            nav_frame = result.nav_comparison.copy()
            if result.benchmark_label and not result.benchmark_nav.empty:
                nav_frame[result.benchmark_label] = result.benchmark_nav.reindex(nav_frame.index).ffill()
            _render_line_chart(nav_frame)
            st.subheader('Drawdown comparison')
            drawdown_frame = result.drawdown_comparison.copy()
            if result.benchmark_label and not result.benchmark_drawdown.empty:
                drawdown_frame[result.benchmark_label] = result.benchmark_drawdown.reindex(drawdown_frame.index).ffill()
            _render_line_chart(drawdown_frame)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'scenario_key': result.scenario_key, 'weight_smoothing_alpha': result.request.weight_smoothing_alpha, 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Tuning' and service_name == 'Hyperparameter tuning':
    strategy_defaults = STRATEGY_OPTIONS
    hyper_state_key = 'tuning::hyperparameter::result'
    st.info('This is the broadest search view. It explores a grid of strategies, cleaning methods, covariance windows and rebalance frequencies, then returns a ranked results table rather than a single primary NAV story.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategies = _multi_strategy_family_selector(
            default_values=strategy_defaults,
            family_key='hyperparameter::strategy_families',
            strategies_key='hyperparameter::strategies',
        )
        st.caption('Treat this section as the strategy search space. A narrower selection keeps the grid faster to run and easier to interpret.')
    st.markdown('### Service parameters')
    with st.form('hyperparameter_tuning_form'):
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
        window_defaults, num_assets = _universe_covariance_window_range_default(universe)
        window_default = ','.join(str(item) for item in window_defaults)
        _sync_widget_default('hyperparameter::windows', window_default, context=(universe, 'hyperparameter', num_assets))
        smoothing_default = float(config_defaults.get('backtest', {}).get('weight_smoothing_alpha', 1.0) or 1.0)
        st.markdown('#### Search space')
        row1 = st.columns(3)
        with row1[0]:
            methods = st.multiselect(
                'Cleaning methods',
                CLEANING_OPTIONS,
                default=CLEANING_OPTIONS,
                key='hyperparameter::methods',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='hyperparameter::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            windows = st.text_input('Covariance windows', value=str(st.session_state['hyperparameter::windows']), key='hyperparameter::windows')

        st.markdown('#### Backtest context')
        row2 = st.columns(2)
        with row2[0]:
            frequencies = st.multiselect(
                'Rebalance frequencies',
                FREQUENCY_OPTIONS,
                default=FREQUENCY_OPTIONS,
                key='hyperparameter::frequencies',
            )
        with row2[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='hyperparameter::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )

        row3 = st.columns(1)
        with row3[0]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/hyperparameter_tuning', key='hyperparameter::output_dir')
        window_count = max(1, len([item for item in windows.split(',') if item.strip()]))
        _render_service_guidance(
            defaults=f"rebalance frequency={freq_default}, covariance windows={window_default}, weight smoothing alpha={smoothing_default}",
            recommendation='Start with a narrow grid to validate the direction of the search. Widen the space only once the first ranking looks sensible.',
            action=(
                f'evaluate a grid over {max(1, len(strategies))} strategies, {max(1, len(methods))} cleaning methods, '
                f'{window_count} covariance windows and {max(1, len(frequencies))} rebalance frequencies.'
            ),
        )
        st.caption(f'Universe suggestion: {window_default} ({num_assets} assets, rounded up to the nearest 5, with 252 added when needed)')
        run_clicked = st.form_submit_button('Run hyperparameter tuning')
    if run_clicked:
        request = HyperparameterTuningRequest(
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            frequencies=frequencies,
            strategies=strategies,
            methods=methods,
            linear_shrinkage=linear_shrinkage,
            windows=[int(item.strip()) for item in windows.split(',') if item.strip()],
            weight_smoothing_alpha=weight_smoothing_alpha,
            refresh_policy=_consume_refresh_policy(),
            output_dir=output_dir or None,
        )
        st.session_state[hyper_state_key] = run_hyperparameter_tuning(request)
    result = st.session_state.get(hyper_state_key)
    if result is not None:
        summary_tab, config_tab, artifacts_tab = st.tabs(['Summary', 'Config', 'Artifacts'])
        with summary_tab:
            st.subheader('Results table')
            _render_hyperparameter_results_table(result.results_table)
            st.subheader('Skipped configs')
            _render_compact_table(result.skipped_configs, priority=['strategy', 'method', 'covariance_window', 'rebalance_frequency', 'num_assets', 'reason'], empty_message='No skipped configurations.')
            st.subheader('Highlights')
            st.json(result.highlights)
        with config_tab:
            _request_block(result.request, config_defaults, {'universe': result.universe, 'num_scenarios': int(len(result.results_table)), 'skipped_configs': int(len(result.skipped_configs)), 'weight_smoothing_alpha': result.request.weight_smoothing_alpha, 'visible_columns': list(_prepare_table(result.results_table, priority=['strategy', 'method', 'covariance_window', 'rebalance_frequency']).columns), 'highlights': result.highlights})
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

elif usage_mode == 'Inspection' and service_name == 'Inspection snapshot':
    strategy_default = config_defaults.get('evaluation', {}).get('strategy', STRATEGY_OPTIONS[0])
    inspection_state_key = 'inspection::snapshot::result'
    st.info('Inspection snapshot is a diagnostic view of one dated state. It is different from evaluation because it focuses on matrices, spectra, features and one portfolio snapshot rather than a full performance path.')
    st.markdown('### Strategy')
    with st.container(border=True):
        strategy = _single_strategy_family_selector(
            default_value=strategy_default,
            family_key='inspection::snapshot::strategy_family',
            strategy_key='inspection::snapshot::strategy',
        )
    st.markdown('### Service parameters')
    with st.form('inspection_snapshot_form'):
        cleaning_default = config_defaults.get('estimation', {}).get('cleaning_method', CLEANING_OPTIONS[0])
        shrinkage_default = float(config_defaults.get('estimation', {}).get('linear_shrinkage', 0.0) or 0.0)
        freq_default = config_defaults.get('evaluation', {}).get('rebalance_frequency', FREQUENCY_OPTIONS[0])
        long_only_default = bool(config_defaults.get('backtest', {}).get('long_only', False))
        smoothing_default = float(config_defaults.get('backtest', {}).get('weight_smoothing_alpha', 1.0) or 1.0)
        window_default, num_assets = _universe_covariance_window_default(universe)
        _sync_widget_default('inspection::snapshot::covariance_window', window_default, context=(universe, 'inspection_snapshot', num_assets))
        inspection_date_default = config_defaults.get('allocation', {}).get('date')

        row1 = st.columns(3)
        with row1[0]:
            cleaning_method = st.selectbox(
                'Cleaning method',
                CLEANING_OPTIONS,
                index=CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in CLEANING_OPTIONS else 0,
                key='inspection::snapshot::cleaning_method',
            )
        with row1[1]:
            linear_shrinkage = _linear_shrinkage_input(
                key='inspection::snapshot::linear_shrinkage',
                default_value=shrinkage_default,
            )
        with row1[2]:
            covariance_window = int(
                st.number_input(
                    'Covariance window',
                    min_value=2,
                    value=int(st.session_state['inspection::snapshot::covariance_window']),
                    step=1,
                    key='inspection::snapshot::covariance_window',
                )
            )

        row2 = st.columns(3)
        with row2[0]:
            rebalance_frequency = st.selectbox(
                'Rebalance frequency',
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(freq_default) if freq_default in FREQUENCY_OPTIONS else 0,
                key='inspection::snapshot::rebalance_frequency',
            )
        with row2[1]:
            weight_smoothing_alpha = float(
                st.number_input(
                    'Weight smoothing alpha',
                    min_value=0.0,
                    max_value=1.0,
                    value=smoothing_default,
                    step=0.05,
                    key='inspection::snapshot::weight_smoothing_alpha',
                    help='1.0 = no smoothing. Lower values reduce turnover and slow reallocation.',
                )
            )
        with row2[2]:
            long_only = st.checkbox('Long only', value=long_only_default, key='inspection::snapshot::long_only')

        row3 = st.columns(3)
        with row3[0]:
            use_latest_inspection = st.checkbox(
                'Use latest available inspection date',
                value=inspection_date_default in (None, '', 'None'),
                key='inspection::snapshot::latest_date',
            )
        with row3[1]:
            inspection_date_selected = st.date_input(
                'Inspection date',
                value=_parse_default_date(inspection_date_default).date(),
                key='inspection::snapshot::date',
                disabled=use_latest_inspection,
            )
            inspection_date = None if use_latest_inspection else pd.Timestamp(inspection_date_selected).date().isoformat()
        with row3[2]:
            output_dir = st.text_input('Output dir', value='output/optimal_tf/dashboard/inspection_snapshot', key='inspection::snapshot::output_dir')
        _render_service_guidance(
            defaults=(
                f"strategy={strategy_default}, cleaning={cleaning_default}, covariance window={window_default}, "
                f"rebalance frequency={freq_default}, weight smoothing alpha={smoothing_default}"
            ),
            recommendation='Use the latest inspection date for an operational diagnostic, or pick a manual date when you want to understand a historical rebalance decision.',
            action=f'build one inspection snapshot for strategy `{strategy}` on `{inspection_date or "the latest available date"}`.',
        )
        st.caption(f'Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)')
        run_clicked = st.form_submit_button('Run inspection snapshot')
    if run_clicked:
        request = InspectionSnapshotRequest(
            refresh_policy=_consume_refresh_policy(),
            config_path=config_path_input,
            universe=universe,
            start=start or None,
            evaluation_start=common_evaluation_start or None,
            evaluation_end=common_evaluation_end or None,
            rebalance_frequency=rebalance_frequency,
            strategy=strategy,
            date=inspection_date,
            cleaning_method=cleaning_method,
            linear_shrinkage=linear_shrinkage,
            covariance_window=int(covariance_window),
            weight_smoothing_alpha=weight_smoothing_alpha,
            long_only=long_only,
            output_dir=output_dir or None,
        )
        st.session_state[inspection_state_key] = run_inspection_snapshot(request)
    result = st.session_state.get(inspection_state_key)
    if result is not None:
        results_tab, config_tab, artifacts_tab = st.tabs(['Results', 'Config', 'Artifacts'])
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
                st.subheader('Cleaner diagnostic vs empirical')
                _render_compact_table(
                    result.cleaner_comparison_frame,
                    priority=[
                        'reference_cleaner',
                        'selected_cleaner',
                        'linear_shrinkage',
                        'max_abs_corr_diff_vs_empirical',
                        'mean_abs_corr_diff_vs_empirical',
                        'max_abs_weight_diff_vs_empirical',
                        'mean_abs_weight_diff_vs_empirical',
                        'l1_weight_diff_vs_empirical',
                        'signal_scale_vs_empirical',
                        'empirical_signal_scale',
                    ],
                )
                st.subheader('Portfolio diagnostic vs empirical')
                _render_compact_table(
                    result.portfolio_comparison_frame,
                    priority=[
                        'selected_cleaner',
                        'reference_cleaner',
                        'selected_total_return',
                        'empirical_total_return',
                        'total_return_diff_vs_empirical',
                        'selected_sharpe',
                        'empirical_sharpe',
                        'sharpe_diff_vs_empirical',
                        'selected_avg_turnover',
                        'empirical_avg_turnover',
                        'avg_turnover_diff_vs_empirical',
                        'daily_return_corr_vs_empirical',
                        'tracking_error_ann_vs_empirical',
                        'max_abs_daily_return_diff_vs_empirical',
                        'mean_abs_daily_return_diff_vs_empirical',
                    ],
                )
                st.subheader('Portfolio NAV vs empirical')
                _render_line_chart(result.portfolio_nav_comparison)
            with matrices_tab:
                st.subheader('Sample correlation heatmap')
                _render_matrix_heatmap(result.sample_correlation, title='Sample correlation')
                st.subheader('Empirical cleaned correlation heatmap')
                _render_matrix_heatmap(result.empirical_cleaned_correlation, title='Empirical cleaned correlation')
                st.subheader('Cleaned correlation heatmap')
                _render_matrix_heatmap(result.cleaned_correlation, title='Cleaned correlation')
                st.subheader('Cleaned covariance heatmap')
                _render_matrix_heatmap(result.cleaned_covariance, title='Cleaned covariance', cmap='viridis')
                left, center, right = st.columns(3)
                with left:
                    st.caption('Sample correlation preview')
                    _render_colored_frame(result.sample_correlation, max_rows=30, max_cols=30)
                with center:
                    st.caption('Empirical cleaned correlation preview')
                    _render_colored_frame(result.empirical_cleaned_correlation, max_rows=30, max_cols=30)
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
                st.subheader('Selected cleaner allocation snapshot')
                _render_colored_frame(result.allocation_frame, max_rows=200, max_cols=result.allocation_frame.shape[1], cmap='RdBu_r')
                st.subheader('Empirical allocation snapshot')
                _render_colored_frame(result.empirical_allocation_frame, max_rows=200, max_cols=result.empirical_allocation_frame.shape[1], cmap='RdBu_r')
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
                'weight_smoothing_alpha': result.request.weight_smoothing_alpha,
                'cleaner_comparison': result.cleaner_comparison_frame.iloc[0].to_dict(),
                'portfolio_comparison': result.portfolio_comparison_frame.iloc[0].to_dict(),
                'inspection_rebalance_frequency': rebalance_frequency,
                'evaluation_start': common_evaluation_start,
                'evaluation_end': common_evaluation_end,
            })
        with artifacts_tab:
            _artifacts_block(result.artifacts.files)

else:
    st.info('This inspection UI now focuses on the snapshot workflow only.')
