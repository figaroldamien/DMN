from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from market_tickers_data import MARKET_TICKERS
from market_tickers_data.universes import (
    CAC40_COMPONENTS,
    DATASET_COMPONENTS,
    DJI_COMPONENTS,
    EUROSTOXX50_COMPONENTS,
    EUROSTOXX600_COMPONENTS,
    FUTURES_COMPONENTS,
    INDEX_COMPONENTS,
    NASDAQ100_COMPONENTS,
    SBF120_COMPONENTS,
    SP500_COMPONENTS,
    WORLD_INDEX_COMPONENTS,
)
from optimal_tf.config_io import load_config
from optimal_tf.dashboard_workspace import (
    build_workspace_context,
    normalize_workspace_selection,
    workspace_defaults_from_config,
)
from optimal_tf.data import load_prices_for_universe, load_prices_yf
from optimal_tf.rebalance import supported_rebalance_frequencies
from optimal_tf.services import (
    InspectionIntervalRequest,
    InspectionSnapshotRequest,
    run_inspection_interval,
    run_inspection_snapshot,
)
from trading_core.features import alpha_from_span
from trading_core.market.universes import get_universe_benchmark
from trading_core.reporting import (
    cumulative_nav,
    equal_weight_rebalanced_benchmark,
    single_asset_buy_and_hold_benchmark,
)
from trading_core.reporting.plots import plt
from trading_core.risk import marchenko_pastur_law

DEFAULT_CONFIG = "configs/matrix_inspection.toml"
PRODUCT_MODES = ("Workspace", "Inspection")
MODE_SERVICES = {
    "Workspace": {
        "Config editor": "Edit the TOML configuration used by the matrix inspection app.",
    },
    "Inspection": {
        "Inspect at date": "Inspect one dated cleaned-matrix state with spectra, eigenvectors and cross-asset features.",
        "Inspect over interval": "Inspect how cleaned matrices and leading eigenmodes evolve over an interval of rebalance dates.",
    },
}
UNIVERSE_OPTIONS = sorted(MARKET_TICKERS)
MARKET_UNIVERSES = ["cac40", "dji", "eurostoxx50", "eurostoxx600", "nasdaq100", "sbf120", "sp500"]
INDEX_UNIVERSES = ["dataset_all", "futures", "index", "table8_all", "world_index", "test"]
UNIVERSE_GROUPS = {
    "Markets": [name for name in MARKET_UNIVERSES if name in MARKET_TICKERS],
    "Index universes": [name for name in INDEX_UNIVERSES if name in MARKET_TICKERS],
}
UNIVERSE_COMPONENTS = {
    "nasdaq100": NASDAQ100_COMPONENTS,
    "cac40": CAC40_COMPONENTS,
    "dji": DJI_COMPONENTS,
    "eurostoxx50": EUROSTOXX50_COMPONENTS,
    "eurostoxx600": EUROSTOXX600_COMPONENTS,
    "sbf120": SBF120_COMPONENTS,
    "sp500": SP500_COMPONENTS,
    "index": INDEX_COMPONENTS,
    "futures": FUTURES_COMPONENTS,
    "dataset": DATASET_COMPONENTS,
    "dataset_all": DATASET_COMPONENTS,
    "table8_all": DATASET_COMPONENTS,
    "table_8": DATASET_COMPONENTS,
    "world_index": WORLD_INDEX_COMPONENTS,
}
FREQUENCY_OPTIONS = supported_rebalance_frequencies()
MATRIX_INPUT_OPTIONS = ["normalized_returns", "raw_returns"]
MATRIX_INPUT_LABELS = {
    "normalized_returns": "Normalized returns",
    "raw_returns": "Raw returns",
}
MATRIX_TYPE_OPTIONS = ["correlation", "covariance"]
MATRIX_TYPE_LABELS = {
    "correlation": "Correlation",
    "covariance": "Covariance",
}
MATRIX_ESTIMATOR_OPTIONS = ["sample_window", "ewma"]
MATRIX_ESTIMATOR_LABELS = {
    "sample_window": "Sample window",
    "ewma": "EWMA",
}
MATRIX_INSPECTION_CLEANING_OPTIONS = [
    "empirical",
    "rie_spectral",
    "rie_reference",
    "linear_shrinkage",
]
MAX_CHART_POINTS = 750
REFRESH_NEXT_RUN_KEY = "matrix_inspection::refresh_next_run"

st.set_page_config(page_title="matrix inspection dashboard", layout="wide")
st.title("matrix inspection dashboard")
st.caption("App autonome pour inspecter les matrices nettoyees et leur structure spectrale avec les services `optimal_tf`.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return value.rename("value").reset_index().to_dict(orient="records")
    if isinstance(value, pd.DataFrame):
        return value.head(200).to_dict(orient="records")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise ValueError("None values must be handled before TOML serialization.")
    if isinstance(value, (int, float)):
        return repr(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_array(values: list[Any]) -> str:
    return "[" + ", ".join(_toml_scalar(value) for value in values) + "]"


def _build_config_toml(payload: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    section_order = ["universe", "estimation", "evaluation", "inspection"]
    for section in section_order:
        values = payload.get(section, {})
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple)):
                lines.append(f"{key} = {_toml_array(list(value))}")
            else:
                lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _read_toml_mapping(path: str | Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _drop_empty_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        values = [value for value in row.values() if value not in (None, "")]
        if values:
            filtered.append(row)
    return filtered


def _load_defaults(config_path: str) -> tuple[dict[str, Any], str | None]:
    try:
        raw = _read_toml_mapping(config_path)
        universe, estimation, backtest, allocation, evaluation, compare, output = load_config(config_path)
    except Exception as exc:  # pragma: no cover
        return {}, str(exc)
    return {
        "universe": asdict(universe),
        "estimation": asdict(estimation),
        "backtest": asdict(backtest),
        "allocation": asdict(allocation),
        "evaluation": asdict(evaluation),
        "compare": asdict(compare),
        "output": asdict(output),
        "inspection": raw.get("inspection", {}) if isinstance(raw.get("inspection"), dict) else {},
    }, None


def _workspace_overview_rows(config_path: str, config_defaults: dict[str, Any]) -> list[dict[str, str]]:
    universe_defaults = config_defaults.get("universe", {})
    evaluation_defaults = config_defaults.get("evaluation", {})
    estimation_defaults = config_defaults.get("estimation", {})
    return [
        {"field": "config_path", "value": str(config_path)},
        {"field": "universe", "value": str(universe_defaults.get("name", ""))},
        {"field": "start", "value": str(universe_defaults.get("start", ""))},
        {"field": "cleaning_method", "value": str(estimation_defaults.get("cleaning_method", ""))},
        {"field": "covariance_window", "value": str(estimation_defaults.get("covariance_window", ""))},
        {"field": "rebalance_frequency", "value": str(evaluation_defaults.get("rebalance_frequency", ""))},
        {"field": "snapshot_output_dir", "value": str(config_defaults.get("inspection", {}).get("snapshot_output_dir", ""))},
        {"field": "interval_output_dir", "value": str(config_defaults.get("inspection", {}).get("interval_output_dir", ""))},
        {
            "field": "evaluation_window",
            "value": " -> ".join(
                [item for item in [str(evaluation_defaults.get("evaluation_start", "") or ""), str(evaluation_defaults.get("evaluation_end", "") or "")] if item]
            ),
        },
    ]


def _format_universe_label(value: str) -> str:
    labels = {
        "cac40": "CAC 40",
        "dji": "DJI",
        "eurostoxx50": "EURO STOXX 50",
        "eurostoxx600": "EURO STOXX 600",
        "nasdaq100": "NASDAQ 100",
        "sbf120": "SBF 120",
        "sp500": "S&P 500",
        "dataset_all": "Dataset all",
        "futures": "Futures",
        "index": "Index",
        "table8_all": "Table 8 all",
        "world_index": "World index",
        "test": "Test",
    }
    return labels.get(value, value)


def _handle_navigation_change() -> None:
    st.session_state["matrix_inspection::pending_signature"] = (
        st.session_state.get("matrix_inspection::usage_mode", "Workspace"),
        st.session_state.get("matrix_inspection::service_name", "Config editor"),
    )


def _mode_service_selector() -> tuple[str, str]:
    usage_mode = st.sidebar.radio(
        "Usage mode",
        PRODUCT_MODES,
        key="matrix_inspection::usage_mode",
        on_change=_handle_navigation_change,
    )
    catalog = MODE_SERVICES[usage_mode]
    service_options = list(catalog.keys())
    if usage_mode == "Inspection":
        preferred_service = st.session_state.get("matrix_inspection::default_service_from_config")
        if preferred_service in service_options and st.session_state.get("matrix_inspection::service_name") not in service_options:
            st.session_state["matrix_inspection::service_name"] = preferred_service
    current_service = st.session_state.get("matrix_inspection::service_name")
    if current_service not in service_options:
        st.session_state["matrix_inspection::service_name"] = service_options[0]
    service_name = st.sidebar.selectbox(
        "Service",
        service_options,
        key="matrix_inspection::service_name",
        on_change=_handle_navigation_change,
    )
    st.sidebar.caption(catalog[service_name])
    return usage_mode, service_name


def _inspection_defaults(config_defaults: dict[str, Any]) -> dict[str, Any]:
    defaults = config_defaults.get("inspection", {})
    return defaults if isinstance(defaults, dict) else {}


def _parse_default_date(default_value: Any) -> pd.Timestamp:
    if default_value in (None, "", "None"):
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(default_value).normalize()


def _date_input_value(label: str, default_value: Any, *, key: str) -> str:
    selected = st.sidebar.date_input(label, value=_parse_default_date(default_value).date(), key=key)
    return pd.Timestamp(selected).date().isoformat()


def _linear_shrinkage_input(*, key: str, default_value: float) -> float:
    return float(
        st.number_input(
            "Linear shrinkage",
            min_value=0.0,
            max_value=1.0,
            value=float(default_value),
            step=0.05,
            key=key,
        )
    )


def _render_config_editor(config_path: str, config_defaults: dict[str, Any]) -> None:
    st.markdown(
        "This workspace page defines the shared defaults used by the matrix inspection services. "
        "Use it when you want to change the default market context or persistent inspection assumptions."
    )
    st.info(
        "Use this page when you want to change persistent defaults. "
        "Use `Inspection` when you want one-off execution under the current workspace."
    )
    with st.container(border=True):
        st.caption("Current workspace")
        _render_compact_table(
            pd.DataFrame(_drop_empty_rows(_workspace_overview_rows(config_path, config_defaults))),
            priority=["field", "value"],
            empty_message="No workspace defaults available.",
        )
    st.caption(f"Editing: `{config_path}`")
    default_save_path = st.session_state.get("matrix_inspection::config::save_path", config_path)

    universe_defaults = config_defaults.get("universe", {})
    estimation_defaults = config_defaults.get("estimation", {})
    evaluation_defaults = config_defaults.get("evaluation", {})
    inspection_defaults = config_defaults.get("inspection", {})

    with st.form("matrix_inspection_config_editor_form"):
        st.markdown("### Essential defaults")
        essential_row1 = st.columns(2)
        with essential_row1[0]:
            universe_name = st.selectbox(
                "Universe",
                UNIVERSE_OPTIONS,
                index=UNIVERSE_OPTIONS.index(universe_defaults.get("name", UNIVERSE_OPTIONS[0])) if universe_defaults.get("name") in UNIVERSE_OPTIONS else 0,
            )
        with essential_row1[1]:
            universe_start = st.text_input("Start date", value=str(universe_defaults.get("start", "2000-01-01") or "2000-01-01"))

        essential_row2 = st.columns(3)
        with essential_row2[0]:
            cleaning_method = st.selectbox(
                "Cleaning method",
                MATRIX_INSPECTION_CLEANING_OPTIONS,
                index=MATRIX_INSPECTION_CLEANING_OPTIONS.index(estimation_defaults.get("cleaning_method", MATRIX_INSPECTION_CLEANING_OPTIONS[0]))
                if estimation_defaults.get("cleaning_method") in MATRIX_INSPECTION_CLEANING_OPTIONS
                else 0,
            )
        with essential_row2[1]:
            covariance_window = int(
                st.number_input("Covariance window", min_value=2, value=int(estimation_defaults.get("covariance_window", 150) or 150), step=1)
            )
        with essential_row2[2]:
            linear_shrinkage = float(
                st.number_input("Linear shrinkage", min_value=0.0, max_value=1.0, value=float(estimation_defaults.get("linear_shrinkage", 0.0) or 0.0), step=0.05)
            )

        essential_row3 = st.columns(3)
        with essential_row3[0]:
            rebalance_frequency = st.selectbox(
                "Rebalance frequency",
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(evaluation_defaults.get("rebalance_frequency", FREQUENCY_OPTIONS[0]))
                if evaluation_defaults.get("rebalance_frequency") in FREQUENCY_OPTIONS
                else 0,
            )
        with essential_row3[1]:
            evaluation_start = st.text_input("Evaluation start", value=str(evaluation_defaults.get("evaluation_start", "") or ""))
        with essential_row3[2]:
            evaluation_end = st.text_input("Evaluation end", value=str(evaluation_defaults.get("evaluation_end", "") or ""))

        st.markdown("### App defaults")
        app_row1 = st.columns(3)
        with app_row1[0]:
            snapshot_matrix_type = st.selectbox(
                "Snapshot matrix type",
                MATRIX_TYPE_OPTIONS,
                index=MATRIX_TYPE_OPTIONS.index(str(inspection_defaults.get("snapshot_matrix_type", "correlation")))
                if str(inspection_defaults.get("snapshot_matrix_type", "correlation")) in MATRIX_TYPE_OPTIONS
                else 0,
            )
        with app_row1[1]:
            snapshot_input_type = st.selectbox(
                "Snapshot input type",
                MATRIX_INPUT_OPTIONS,
                index=MATRIX_INPUT_OPTIONS.index(str(inspection_defaults.get("snapshot_input_type", "normalized_returns")))
                if str(inspection_defaults.get("snapshot_input_type", "normalized_returns")) in MATRIX_INPUT_OPTIONS
                else 0,
            )
        with app_row1[2]:
            snapshot_estimator_method = st.selectbox(
                "Snapshot estimator",
                MATRIX_ESTIMATOR_OPTIONS,
                index=MATRIX_ESTIMATOR_OPTIONS.index(str(inspection_defaults.get("snapshot_estimator_method", "sample_window")))
                if str(inspection_defaults.get("snapshot_estimator_method", "sample_window")) in MATRIX_ESTIMATOR_OPTIONS
                else 0,
            )

        app_row2 = st.columns(3)
        with app_row2[0]:
            interval_matrix_type = st.selectbox(
                "Interval matrix type",
                MATRIX_TYPE_OPTIONS,
                index=MATRIX_TYPE_OPTIONS.index(str(inspection_defaults.get("interval_matrix_type", "correlation")))
                if str(inspection_defaults.get("interval_matrix_type", "correlation")) in MATRIX_TYPE_OPTIONS
                else 0,
            )
        with app_row2[1]:
            interval_input_type = st.selectbox(
                "Interval input type",
                MATRIX_INPUT_OPTIONS,
                index=MATRIX_INPUT_OPTIONS.index(str(inspection_defaults.get("interval_input_type", "normalized_returns")))
                if str(inspection_defaults.get("interval_input_type", "normalized_returns")) in MATRIX_INPUT_OPTIONS
                else 0,
            )
        with app_row2[2]:
            interval_estimator_method = st.selectbox(
                "Interval estimator",
                MATRIX_ESTIMATOR_OPTIONS,
                index=MATRIX_ESTIMATOR_OPTIONS.index(str(inspection_defaults.get("interval_estimator_method", "sample_window")))
                if str(inspection_defaults.get("interval_estimator_method", "sample_window")) in MATRIX_ESTIMATOR_OPTIONS
                else 0,
            )

        app_row3 = st.columns(3)
        with app_row3[0]:
            default_snapshot_date = st.text_input(
                "Default snapshot date",
                value=str(inspection_defaults.get("snapshot_date", "") or ""),
                help="Leave empty to default to latest available date.",
            )
        with app_row3[1]:
            snapshot_output_dir = st.text_input(
                "Snapshot output dir",
                value=str(inspection_defaults.get("snapshot_output_dir", "output/matrix_inspection/snapshot") or ""),
            )
        with app_row3[2]:
            interval_output_dir = st.text_input(
                "Interval output dir",
                value=str(inspection_defaults.get("interval_output_dir", "output/matrix_inspection/interval") or ""),
            )

        app_row4 = st.columns(2)
        with app_row4[0]:
            interval_leading_eigenvectors = int(
                st.number_input(
                    "Default leading eigenvectors",
                    min_value=1,
                    max_value=12,
                    value=int(inspection_defaults.get("leading_eigenvectors", 3) or 3),
                    step=1,
                )
            )
        with app_row4[1]:
            default_service = st.selectbox(
                "Default inspection service",
                ["Inspect at date", "Inspect over interval"],
                index=["Inspect at date", "Inspect over interval"].index(
                    str(inspection_defaults.get("default_service", "Inspect at date"))
                )
                if str(inspection_defaults.get("default_service", "Inspect at date")) in {"Inspect at date", "Inspect over interval"}
                else 0,
            )

        with st.expander("Advanced engine defaults", expanded=False):
            adv_left, adv_right = st.columns(2)
            with adv_left:
                vol_span = int(st.number_input("Vol span", min_value=2, value=int(estimation_defaults.get("vol_span", 60) or 60), step=1))
                covariance_min_periods = int(
                    st.number_input("Covariance min periods", min_value=1, value=int(estimation_defaults.get("covariance_min_periods", 60) or 60), step=1)
                )
                max_abs_return = float(st.number_input("Max abs return", min_value=0.0, value=float(estimation_defaults.get("max_abs_return", 1.0) or 1.0), step=0.1))
                rie_bandwidth = float(
                    st.number_input("RIE bandwidth", min_value=0.0, value=float(estimation_defaults.get("rie_bandwidth", 0.001) or 0.001), step=0.0005, format="%.6f")
                )
            with adv_right:
                quality_min_history_days = int(
                    st.number_input("Quality min history days", min_value=1, value=int(universe_defaults.get("quality_min_history_days", 756) or 756), step=1)
                )
                quality_min_coverage_ratio = float(
                    st.number_input("Quality min coverage ratio", min_value=0.0, max_value=1.0, value=float(universe_defaults.get("quality_min_coverage_ratio", 0.9) or 0.9), step=0.05)
                )
                quality_max_internal_missing = int(
                    st.number_input("Quality max internal missing", min_value=0, value=int(universe_defaults.get("quality_max_internal_missing", 0) or 0), step=1)
                )
                quality_require_latest_price = st.checkbox(
                    "Quality require latest price",
                    value=bool(universe_defaults.get("quality_require_latest_price", True)),
                )

        save_col, path_col = st.columns([1, 3])
        with save_col:
            save_clicked = st.form_submit_button("Save config")
        with path_col:
            save_path = st.text_input("Save config as", value=str(default_save_path))

    if save_clicked:
        st.session_state["matrix_inspection::config::save_path"] = save_path
        payload = {
            "universe": {
                "name": universe_name,
                "start": universe_start,
                "quality_filter_enabled": bool(universe_defaults.get("quality_filter_enabled", True)),
                "quality_min_history_days": quality_min_history_days,
                "quality_min_coverage_ratio": quality_min_coverage_ratio,
                "quality_max_internal_missing": quality_max_internal_missing,
                "quality_max_abs_return": max_abs_return,
                "quality_require_latest_price": quality_require_latest_price,
            },
            "estimation": {
                "vol_span": vol_span,
                "covariance_window": covariance_window,
                "covariance_min_periods": covariance_min_periods,
                "max_abs_return": max_abs_return,
                "cleaning_method": cleaning_method,
                "linear_shrinkage": linear_shrinkage,
                "rie_bandwidth": rie_bandwidth,
                "trend_alpha": float(estimation_defaults.get("trend_alpha", 0.01575) or 0.01575),
                "lltf_l2_reg": float(estimation_defaults.get("lltf_l2_reg", 0.0001) or 0.0001),
            },
            "evaluation": {
                "rebalance_frequency": rebalance_frequency,
                "evaluation_start": evaluation_start,
                "evaluation_end": evaluation_end,
            },
            "inspection": {
                "default_service": default_service,
                "snapshot_date": default_snapshot_date,
                "snapshot_matrix_type": snapshot_matrix_type,
                "snapshot_input_type": snapshot_input_type,
                "snapshot_estimator_method": snapshot_estimator_method,
                "snapshot_output_dir": snapshot_output_dir,
                "interval_matrix_type": interval_matrix_type,
                "interval_input_type": interval_input_type,
                "interval_estimator_method": interval_estimator_method,
                "interval_output_dir": interval_output_dir,
                "leading_eigenvectors": interval_leading_eigenvectors,
            },
        }
        destination = Path(save_path).expanduser()
        if not destination.suffix:
            destination = destination.with_suffix(".toml")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_build_config_toml(payload), encoding="utf-8")
        st.success(f"Config saved to `{destination}`.")
        st.rerun()


def _round_up_to_half_ten(value: float) -> int:
    return max(5, int(5 * np.ceil(float(value) / 5.0)))


def _universe_covariance_window_default(universe: str) -> tuple[int, int]:
    num_assets = len(MARKET_TICKERS.get(universe, []))
    return _round_up_to_half_ten(1.5 * max(1, num_assets)), num_assets


def _queue_force_refresh() -> None:
    st.session_state[REFRESH_NEXT_RUN_KEY] = True


def _consume_refresh_policy() -> str:
    if st.session_state.pop(REFRESH_NEXT_RUN_KEY, False):
        return "always"
    return "auto"


def _render_compact_table(
    frame: pd.DataFrame,
    *,
    priority: list[str] | None = None,
    max_rows: int | None = 200,
    empty_message: str = "No data available.",
) -> None:
    if frame.empty:
        st.info(empty_message)
        return
    table = frame.copy()
    if priority:
        ordered = [column for column in priority if column in table.columns]
        ordered.extend(column for column in table.columns if column not in ordered)
        table = table.loc[:, ordered]
    if max_rows is not None:
        table = table.head(max_rows)
    st.dataframe(table, width="stretch", hide_index=True)


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


def _render_rank_chart_grid(
    frame: pd.DataFrame,
    *,
    x_column: str = "date",
    x_title: str | None = None,
    x_temporal: bool = True,
    value_columns: list[str],
    series_labels: dict[str, str] | None = None,
    title_prefix: str,
    height: int = 220,
) -> None:
    if frame.empty:
        st.info("No data available for this chart.")
        return
    ranks = sorted(int(rank) for rank in pd.to_numeric(frame.get("rank"), errors="coerce").dropna().unique())
    if not ranks:
        st.info("No ranked data available for this chart.")
        return
    for rank in ranks:
        panel = frame.loc[pd.to_numeric(frame["rank"], errors="coerce") == rank, [x_column, *value_columns]].copy()
        if panel.empty:
            continue
        x_values = pd.to_datetime(panel[x_column]) if x_temporal else pd.to_numeric(panel[x_column], errors="coerce")
        plot = panel.drop(columns=x_column)
        plot.index = x_values
        plot = plot.sort_index()
        if series_labels:
            plot = plot.rename(columns=series_labels)
        st.caption(f"{title_prefix} {rank}")
        if x_temporal:
            _render_line_chart(plot, height=height)
            continue
        long_frame = plot.reset_index(names=x_column).melt(id_vars=x_column, var_name="series", value_name="value")
        chart = (
            alt.Chart(long_frame)
            .mark_line(strokeWidth=2.0)
            .encode(
                x=alt.X(f"{x_column}:Q", title=x_title),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("series:N", legend=alt.Legend(title=None, orient="top")),
                tooltip=[
                    alt.Tooltip(f"{x_column}:Q", title=x_title or x_column, format=".0f"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("value:Q", title="Value", format=".6f"),
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
    cmap: str = "RdBu_r",
    empty_message: str = "No data available.",
) -> None:
    if frame.empty:
        st.info(empty_message)
        return
    preview = frame.iloc[:max_rows, :max_cols].copy()
    styled = preview.style.format("{:.4f}").background_gradient(cmap=cmap, axis=None)
    st.dataframe(styled, width="stretch", height=min(720, max(220, 38 + 28 * min(len(preview), 16))))


def _render_matrix_heatmap(
    frame: pd.DataFrame,
    *,
    title: str,
    cmap: str = "RdBu_r",
    universe: str | None = None,
    compact: bool = False,
    annotate_values: bool = False,
) -> None:
    if frame.empty:
        st.info("No matrix available.")
        return
    ordered_frame = frame.copy()
    group_boundaries: list[int] = []
    group_centers: list[float] = []
    group_labels: list[str] = []
    subgroup_boundaries: list[int] = []
    if frame.index.equals(frame.columns):
        components = UNIVERSE_COMPONENTS.get(universe or "", {})
        metadata_rows: list[dict[str, Any]] = []
        for original_pos, ticker in enumerate(frame.index):
            meta = components.get(str(ticker), {})
            sector = str(meta.get("sector", "") or "").strip()
            sub_sector = str(meta.get("sub_sector", "") or "").strip()
            category = str(meta.get("category", "") or "").strip()
            sub_category = str(meta.get("sub_category", "") or "").strip()
            primary_group = sector or category
            secondary_group = sub_sector or sub_category
            metadata_rows.append(
                {
                    "ticker": str(ticker),
                    "primary_group": primary_group,
                    "secondary_group": secondary_group,
                    "group_missing": 1 if not primary_group else 0,
                    "subgroup_missing": 1 if not secondary_group else 0,
                    "original_pos": original_pos,
                }
            )
        if metadata_rows:
            metadata_frame = pd.DataFrame(metadata_rows).sort_values(
                by=["group_missing", "primary_group", "subgroup_missing", "secondary_group", "ticker", "original_pos"],
                kind="stable",
            )
            ordered_labels = metadata_frame["ticker"].tolist()
            ordered_frame = frame.reindex(index=ordered_labels, columns=ordered_labels)
            grouped = metadata_frame.loc[metadata_frame["primary_group"].ne("")].groupby("primary_group", sort=False).size()
            cursor = 0
            for group_label, count in grouped.items():
                start = cursor
                cursor += int(count)
                group_centers.append(start + (count - 1) / 2.0)
                group_labels.append(str(group_label))
                group_boundaries.append(cursor)
            if group_boundaries:
                group_boundaries = group_boundaries[:-1]
            subgrouped = metadata_frame.loc[
                metadata_frame["primary_group"].ne("") & metadata_frame["secondary_group"].ne("")
            ].groupby(["primary_group", "secondary_group"], sort=False).size()
            cursor = 0
            for _group_label, count in subgrouped.items():
                cursor += int(count)
                subgroup_boundaries.append(cursor)
            if subgroup_boundaries:
                subgroup_boundaries = subgroup_boundaries[:-1]
    values = ordered_frame.to_numpy(dtype=float)
    base_size = 0.16 if compact else 0.22
    min_size = 4.6 if compact else 6.0
    max_size = 8.2 if compact else 14.0
    fig_size = max(min_size, min(max_size, base_size * len(frame) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    vmax = max(vmax, 1e-12)
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    tick_target = 12 if compact else 20
    tick_step = max(1, len(ordered_frame) // tick_target)
    positions = np.arange(0, len(ordered_frame), tick_step)
    tick_fontsize = 5.5 if compact else 7
    group_fontsize = 5 if compact else 6
    ax.set_xticks(positions)
    ax.set_xticklabels([str(ordered_frame.columns[pos]) for pos in positions], rotation=90, fontsize=tick_fontsize)
    if group_centers:
        ax.set_yticks(group_centers)
        ax.set_yticklabels(group_labels, fontsize=group_fontsize)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels([str(ordered_frame.index[pos]) for pos in positions], fontsize=tick_fontsize)
    for boundary in group_boundaries:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.4, alpha=0.95)
        ax.axvline(boundary - 0.5, color="black", linewidth=1.4, alpha=0.95)
    for boundary in subgroup_boundaries:
        ax.axhline(boundary - 0.5, color="black", linewidth=0.8, alpha=0.45)
        ax.axvline(boundary - 0.5, color="black", linewidth=0.8, alpha=0.45)
    if annotate_values:
        font_size = 6 if compact else 8
        threshold = 0.55 * vmax
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                text_color = "white" if abs(float(value)) >= threshold else "black"
                ax.text(col_idx, row_idx, f"{float(value):.2f}", ha="center", va="center", fontsize=font_size, color=text_color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _render_mp_outlier_eigenvectors(
    eigenvector_frame: pd.DataFrame,
    spectrum_frame: pd.DataFrame,
    *,
    num_assets: int,
    sample_size: int,
) -> None:
    if eigenvector_frame.empty or spectrum_frame.empty:
        st.info("No eigenvector data available.")
        return
    try:
        mp_law = marchenko_pastur_law(num_assets=num_assets, sample_size=sample_size, variance=1.0)
    except ValueError as exc:
        st.warning(f"Unable to build the Marchenko-Pastur reference law: {exc}")
        return
    outlier_spectrum = spectrum_frame.loc[
        pd.to_numeric(spectrum_frame.get("eigenvalue"), errors="coerce") > float(mp_law.lambda_plus)
    ].copy()
    if outlier_spectrum.empty:
        st.info("No correlation eigenvalue is outside the Marchenko-Pastur bulk on this date.")
        return
    display_eigenvector_frame = eigenvector_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    subplot_count = len(outlier_spectrum)
    fig, axes = plt.subplots(subplot_count, 1, sharex=True, figsize=(18.0, max(3.0 * subplot_count, 3.8)), dpi=180)
    if subplot_count == 1:
        axes = [axes]
    positions = np.arange(len(display_eigenvector_frame.index))
    labels = [str(item[-1]) if isinstance(item, tuple) else str(item) for item in display_eigenvector_frame.index]
    for ax, (_, spectrum_row) in zip(axes, outlier_spectrum.iterrows()):
        rank = int(pd.to_numeric(spectrum_row["rank"], errors="coerce"))
        column_name = f"corr_ev{rank}"
        if column_name not in display_eigenvector_frame.columns:
            continue
        weights = pd.to_numeric(display_eigenvector_frame[column_name], errors="coerce").to_numpy(dtype=float)
        ax.axhline(0.0, color="#666666", linewidth=0.9, alpha=0.7)
        ax.vlines(positions, 0.0, weights, color="#2C6E91", linewidth=1.0, alpha=0.9)
        ax.set_ylabel("Weight")
        ax.set_title(f"Rank {rank} | eigenvalue={float(pd.to_numeric(spectrum_row['eigenvalue'], errors='coerce')):.4f}", loc="left")
        ax.grid(True, axis="y", alpha=0.18)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _render_weight_subplots(weight_frame: pd.DataFrame, spectrum_frame: pd.DataFrame, *, title_prefix: str) -> None:
    if weight_frame.empty:
        st.info("No portfolio weight data available.")
        return
    display_frame = weight_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    spectrum_lookup = spectrum_frame.copy()
    if "rank" in spectrum_lookup.columns:
        spectrum_lookup["rank"] = pd.to_numeric(spectrum_lookup["rank"], errors="coerce")
        spectrum_lookup = spectrum_lookup.dropna(subset=["rank"]).set_index("rank")
    subplot_count = display_frame.shape[1]
    fig, axes = plt.subplots(subplot_count, 1, sharex=True, figsize=(18.0, max(3.0 * subplot_count, 3.8)), dpi=180)
    if subplot_count == 1:
        axes = [axes]
    positions = np.arange(len(display_frame.index))
    labels = [str(item[-1]) if isinstance(item, tuple) else str(item) for item in display_frame.index]
    for ax, column_name in zip(axes, display_frame.columns):
        weights = pd.to_numeric(display_frame[column_name], errors="coerce").to_numpy(dtype=float)
        rank = int(str(column_name).removeprefix("corr_ev")) if str(column_name).startswith("corr_ev") else None
        title = str(column_name)
        if rank is not None and rank in spectrum_lookup.index:
            eigenvalue = float(pd.to_numeric(spectrum_lookup.loc[rank, "eigenvalue"], errors="coerce"))
            title = f"{title_prefix} {rank} | eigenvalue={eigenvalue:.4f}"
        ax.axhline(0.0, color="#666666", linewidth=0.9, alpha=0.7)
        ax.vlines(positions, 0.0, weights, color="#2C6E91", linewidth=1.0, alpha=0.9)
        ax.set_ylabel("Weight")
        ax.set_title(title, loc="left")
        ax.grid(True, axis="y", alpha=0.18)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _render_eigenvalue_distribution_with_mp(
    spectrum_frame: pd.DataFrame,
    *,
    num_assets: int,
    sample_size: int,
) -> None:
    if spectrum_frame.empty:
        st.info("No correlation spectrum available.")
        return
    eigenvalues = pd.to_numeric(spectrum_frame.get("eigenvalue"), errors="coerce").dropna().to_numpy(dtype=float)
    if eigenvalues.size == 0:
        st.info("No correlation eigenvalues available.")
        return
    try:
        mp_law = marchenko_pastur_law(num_assets=num_assets, sample_size=sample_size, variance=1.0)
    except ValueError as exc:
        st.warning(f"Unable to build the Marchenko-Pastur reference law: {exc}")
        return
    grid, density = mp_law.density_grid(num_points=512, padding=0.08)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bin_count = min(36, max(10, int(np.sqrt(eigenvalues.size) * 2.0)))
    ax.hist(eigenvalues, bins=bin_count, density=True, alpha=0.55, color="#4C78A8", edgecolor="white", label="Correlation eigenvalues")
    ax.plot(grid, density, color="#F58518", linewidth=2.0, label="Marchenko-Pastur density")
    ax.axvline(mp_law.lambda_minus, color="#54A24B", linestyle="--", linewidth=1.5, label="MP bulk lower bound")
    ax.axvline(mp_law.lambda_plus, color="#E45756", linestyle="--", linewidth=1.5, label="MP bulk upper bound")
    ax.set_title("Correlation eigenvalue distribution vs Marchenko-Pastur")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _artifacts_block(files: dict[str, Path]) -> None:
    if not files:
        st.info("No artifacts available.")
        return
    rows = [{"name": key, "path": str(value)} for key, value in files.items()]
    _render_compact_table(pd.DataFrame(rows), priority=["name", "path"])


def _request_block(request: Any, config_defaults: dict[str, Any], resolved: dict[str, Any] | None = None) -> None:
    left, center, right = st.columns(3)
    with left:
        st.caption("Request payload")
        st.json(_json_safe(request), expanded=2)
    with center:
        st.caption("Defaults from config")
        st.json(_json_safe(config_defaults), expanded=2)
    with right:
        st.caption("Resolved context")
        st.json(_json_safe(resolved or {}), expanded=2)


def _load_universe_benchmark_nav(
    *,
    config_path: str,
    universe_name: str,
    request_start: str | None,
    target_index: pd.Index,
    refresh_policy: str = "auto",
) -> tuple[str, pd.Series] | None:
    if len(target_index) == 0:
        return None
    try:
        universe_cfg, estimation, *_ = load_config(config_path)
        effective_start = request_start or universe_cfg.start
        prices = load_prices_for_universe(universe_name, start=effective_start, refresh_policy=refresh_policy)
        benchmark = get_universe_benchmark(universe_name)
        if benchmark and benchmark.get("ticker"):
            benchmark_prices = load_prices_yf([str(benchmark["ticker"])], start=effective_start)
            benchmark_returns = single_asset_buy_and_hold_benchmark(
                benchmark_prices,
                max_abs_return=getattr(estimation, "max_abs_return", None),
            )
            benchmark_label = str(benchmark.get("name") or benchmark.get("ticker"))
        else:
            benchmark_returns = equal_weight_rebalanced_benchmark(
                prices,
                max_abs_return=getattr(estimation, "max_abs_return", None),
            )
            benchmark_label = "universe equal-weight index"
        aligned_returns = benchmark_returns.reindex(pd.Index(target_index)).ffill().fillna(0.0)
        return benchmark_label, cumulative_nav(aligned_returns)
    except Exception:
        return None


def _render_snapshot_result(result: Any, *, config_defaults: dict[str, Any]) -> None:
    summary_tab, matrices_tab, spectrum_tab, eigenvectors_tab, config_tab, artifacts_tab = st.tabs(
        ["Summary", "Matrices", "Spectrum", "Eigenvectors", "Config", "Artifacts"]
    )
    with summary_tab:
        st.subheader("Snapshot summary")
        _render_compact_table(
            pd.DataFrame(
                [
                    {
                        "universe": result.universe,
                        "cleaning_method": result.cleaning_method,
                        "matrix_type": MATRIX_TYPE_LABELS.get(result.matrix_type, result.matrix_type),
                        "estimator_method": MATRIX_ESTIMATOR_LABELS.get(result.estimator_method, result.estimator_method),
                        "input_type": MATRIX_INPUT_LABELS.get(result.input_type, result.input_type),
                        "estimator_window": result.estimator_window,
                        "allocation_date": str(result.allocation_date.date()),
                        "num_assets": result.num_assets,
                        "sample_size": result.sample_size,
                    }
                ]
            ),
            priority=["universe", "matrix_type", "estimator_method", "cleaning_method", "input_type", "estimator_window", "allocation_date", "num_assets", "sample_size"],
        )
        st.subheader("Cleaner diagnostic vs empirical")
        _render_compact_table(result.cleaner_comparison_frame)
        st.subheader("Features at inspection date")
        _render_colored_frame(result.feature_frame, max_rows=200, max_cols=result.feature_frame.shape[1], cmap="RdYlBu_r")
    with matrices_tab:
        if result.matrix_type == "covariance":
            sample_matrix = result.sample_covariance
            baseline_matrix = result.empirical_cleaned_covariance
            cleaned_matrix = result.cleaned_covariance
            primary_label = "Covariance"
        else:
            sample_matrix = result.sample_correlation
            baseline_matrix = result.empirical_cleaned_correlation
            cleaned_matrix = result.cleaned_correlation
            primary_label = "Correlation"
        st.subheader(f"Cleaned {primary_label.lower()} heatmap")
        _render_matrix_heatmap(cleaned_matrix, title=f"Cleaned {primary_label.lower()}", universe=result.universe)
        st.subheader("Cleaner effect vs empirical baseline")
        _render_matrix_heatmap(
            cleaned_matrix - baseline_matrix,
            title=f"Cleaned minus empirical baseline {primary_label.lower()}",
            universe=result.universe,
        )
        left, center, right = st.columns(3)
        with left:
            st.caption("Cleaned preview")
            _render_colored_frame(cleaned_matrix, max_rows=30, max_cols=30)
        with center:
            st.caption("Empirical baseline preview")
            _render_colored_frame(baseline_matrix, max_rows=30, max_cols=30)
        with right:
            st.caption("Sample preview")
            _render_colored_frame(sample_matrix, max_rows=30, max_cols=30)
        st.subheader("Sector equal-weight portfolio correlation")
        _render_matrix_heatmap(
            result.sample_sector_ew_correlation,
            title="Sector EW portfolio correlation",
            annotate_values=True,
        )
        with st.expander("Open grouped details", expanded=False):
            group_left, group_center, group_right = st.columns(3)
            with group_left:
                st.caption("Sector membership")
                _render_compact_table(result.sector_membership, priority=["group", "sector", "num_tickers", "ticker"])
            with group_center:
                st.caption("Sector pair counts")
                _render_colored_frame(result.sector_pair_counts, max_rows=40, max_cols=40, cmap="Blues")
            with group_right:
                st.caption("Sub-sector pair counts")
                _render_colored_frame(result.sub_sector_pair_counts, max_rows=40, max_cols=40, cmap="Blues")
    with spectrum_tab:
        spectrum_frame = result.covariance_spectrum if result.matrix_type == "covariance" else result.correlation_spectrum
        empirical_matrix = result.sample_covariance if result.matrix_type == "covariance" else result.sample_correlation
        empirical_eigenvalues = np.linalg.eigvalsh(empirical_matrix.to_numpy(dtype=float))[::-1]
        spectrum_table = spectrum_frame.copy()
        spectrum_table["empirical_eigenvalue"] = empirical_eigenvalues[: len(spectrum_table)]
        st.subheader(f"{MATRIX_TYPE_LABELS.get(result.matrix_type, result.matrix_type)} scree plot")
        scale = st.radio("Eigenvalue scale", ["Linear", "Log"], horizontal=True, key="matrix_inspection::snapshot::spectrum_scale")
        chart = (
            alt.Chart(spectrum_table)
            .mark_line(point=True)
            .encode(
                x=alt.X("rank:Q", title="Rank"),
                y=alt.Y(
                    "eigenvalue:Q",
                    title="Eigenvalue",
                    scale=alt.Scale(type="log") if scale == "Log" else alt.Scale(type="linear"),
                ),
                tooltip=[
                    alt.Tooltip("rank:Q", title="Rank"),
                    alt.Tooltip("eigenvalue:Q", title="Eigenvalue", format=".6f"),
                    alt.Tooltip("empirical_eigenvalue:Q", title="Empirical eigenvalue", format=".6f"),
                    alt.Tooltip("variance_share:Q", title="Variance share", format=".4f"),
                    alt.Tooltip("cumulative_variance_share:Q", title="Cum. variance", format=".4f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, width="stretch")
        _render_compact_table(
            spectrum_table,
            priority=["rank", "eigenvalue", "empirical_eigenvalue", "variance_share", "cumulative_variance_share"],
            max_rows=40,
        )
        if result.matrix_type == "correlation":
            _render_eigenvalue_distribution_with_mp(
                result.correlation_spectrum,
                num_assets=result.num_assets,
                sample_size=result.sample_size,
            )
    with eigenvectors_tab:
        st.subheader("Correlation eigenvectors outside the Marchenko-Pastur bulk")
        _render_mp_outlier_eigenvectors(
            result.correlation_eigenvectors,
            result.correlation_spectrum,
            num_assets=result.num_assets,
            sample_size=result.sample_size,
        )
        st.subheader("Eigenportfolios normalized to sum(weights)=1")
        _render_weight_subplots(
            result.correlation_eigenportfolios,
            result.correlation_spectrum,
            title_prefix="Eigenportfolio",
        )
        st.subheader("NAV of selected eigenportfolios")
        if result.correlation_component_nav.empty or result.correlation_component_nav.shape[1] == 0:
            st.info("No selected eigenportfolio is available on this date.")
        else:
            nav_frame = result.correlation_component_nav
            benchmark_bundle = _load_universe_benchmark_nav(
                config_path=result.request.config_path,
                universe_name=result.universe,
                request_start=result.request.start,
                target_index=nav_frame.index,
                refresh_policy=result.request.refresh_policy,
            )
            fig, axes = plt.subplots(nav_frame.shape[1], 1, sharex=True, figsize=(11, max(3.2 * nav_frame.shape[1], 3.8)), dpi=180)
            if nav_frame.shape[1] == 1:
                axes = [axes]
            for idx, (ax, column_name) in enumerate(zip(axes, nav_frame.columns)):
                series = pd.to_numeric(nav_frame[column_name], errors="coerce")
                ax.plot(series.index, series.to_numpy(dtype=float), color="#2C6E91", linewidth=1.8)
                if idx == 0 and benchmark_bundle is not None:
                    benchmark_label, benchmark_nav = benchmark_bundle
                    ax.plot(benchmark_nav.index, pd.to_numeric(benchmark_nav, errors="coerce").to_numpy(dtype=float), color="#C44E52", linewidth=1.6, alpha=0.9, label=benchmark_label)
                    ax.legend(loc="best")
                ax.set_ylabel("NAV")
                ax.set_title(str(column_name), loc="left", fontsize=12)
                ax.grid(True, alpha=0.22)
            axes[-1].set_xlabel("Date")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            _render_compact_table(
                result.correlation_component_summary,
                priority=["eigenportfolio", "rank", "variance_share", "cumulative_variance_share", "cagr", "ann_vol", "sharpe"],
            )
    with config_tab:
        _request_block(
            result.request,
            config_defaults,
            {
                "universe": result.universe,
                "cleaning_method": result.cleaning_method,
                "matrix_type": result.matrix_type,
                "estimator_method": result.estimator_method,
                "input_type": result.input_type,
                "estimator_window": result.estimator_window,
                "allocation_date": result.allocation_date,
                "sample_size": result.sample_size,
                "num_assets": result.num_assets,
            },
        )
    with artifacts_tab:
        _artifacts_block(result.artifacts.files)


def _render_interval_result(result: Any, *, config_defaults: dict[str, Any]) -> None:
    summary_tab, trends_tab, stability_tab, config_tab, artifacts_tab = st.tabs(
        ["Summary", "Spectrum trends", "Eigenvector stability", "Config", "Artifacts"]
    )
    with summary_tab:
        st.subheader("Interval summary")
        _render_compact_table(
            result.summary_frame,
            priority=["date", "sample_size", "num_assets", "leading_eigenvalue", "second_eigenvalue", "third_eigenvalue", "bulk_outlier_count", "mp_lambda_plus"],
        )
    with trends_tab:
        st.subheader(f"{MATRIX_TYPE_LABELS.get(result.matrix_type, result.matrix_type)} leading eigenvalues over time")
        spectrum_top = result.spectrum_frame[result.spectrum_frame["rank"] <= max(1, int(result.request.leading_eigenvectors))].copy()
        _render_rank_chart_grid(
            spectrum_top,
            value_columns=["eigenvalue"],
            series_labels={"eigenvalue": "eigenvalue"},
            title_prefix="Eigenvalue rank",
        )
        st.subheader("Eigenvalue variogram")
        _render_rank_chart_grid(
            result.variogram_frame,
            x_column="lag",
            x_title="Lag",
            x_temporal=False,
            value_columns=["semivariance"],
            series_labels={"semivariance": "semivariance"},
            title_prefix="Variogram rank",
            height=320,
        )
    with stability_tab:
        st.subheader("Leading eigenvector stability")
        _render_rank_chart_grid(
            result.eigenvector_similarity_frame,
            value_columns=["abs_alignment_anchor", "abs_alignment_previous"],
            series_labels={
                "abs_alignment_anchor": "anchor alignment",
                "abs_alignment_previous": "previous alignment",
            },
            title_prefix="Eigenvector rank",
        )
    with config_tab:
        _request_block(
            result.request,
            config_defaults,
            {
                "universe": result.universe,
                "cleaning_method": result.cleaning_method,
                "matrix_type": result.matrix_type,
                "estimator_method": result.estimator_method,
                "input_type": result.input_type,
                "estimator_window": result.estimator_window,
                "evaluation_start": result.request.evaluation_start,
                "evaluation_end": result.request.evaluation_end,
                "rebalance_frequency": result.request.rebalance_frequency,
                "num_dates": int(len(result.summary_frame)),
                "num_assets": result.num_assets,
            },
        )
    with artifacts_tab:
        _artifacts_block(result.artifacts.files)


usage_mode, service_name = _mode_service_selector()
config_path_input = st.sidebar.text_input("Config path", value=DEFAULT_CONFIG, key="matrix_inspection::config_path")
config_defaults, config_error = _load_defaults(config_path_input)
if config_error:
    st.warning(f"Unable to load config defaults from {config_path_input}: {config_error}")
    config_defaults = {}
inspection_defaults = _inspection_defaults(config_defaults)
st.session_state["matrix_inspection::default_service_from_config"] = str(
    inspection_defaults.get("default_service", "Inspect at date") or "Inspect at date"
)
if (
    usage_mode == "Inspection"
    and st.session_state.get("matrix_inspection::service_name") not in MODE_SERVICES["Inspection"]
):
    st.session_state["matrix_inspection::service_name"] = st.session_state["matrix_inspection::default_service_from_config"]

workspace_defaults = workspace_defaults_from_config(
    config_defaults,
    default_config_path=config_path_input,
    fallback_universe=UNIVERSE_OPTIONS[0],
)
universe = workspace_defaults.universe
start = workspace_defaults.start
evaluation_start = workspace_defaults.evaluation_start
evaluation_end = workspace_defaults.evaluation_end
workspace_universe_group: str | None = None

if usage_mode != "Workspace":
    stored_group, group_options, stored_universe = normalize_workspace_selection(
        universe_groups=UNIVERSE_GROUPS,
        fallback_universe_options=UNIVERSE_OPTIONS,
        universe_default=workspace_defaults.universe,
        stored_group=st.session_state.get("matrix_inspection::universe_group"),
        stored_universe=st.session_state.get("matrix_inspection::universe"),
    )
    group_names = [name for name, options in UNIVERSE_GROUPS.items() if options]
    universe_group = st.sidebar.selectbox(
        "Universe group",
        group_names,
        index=group_names.index(stored_group),
        key="matrix_inspection::universe_group",
    )
    group_options = UNIVERSE_GROUPS.get(universe_group, UNIVERSE_OPTIONS) or UNIVERSE_OPTIONS
    universe_default = stored_universe if stored_universe in group_options else group_options[0]
    universe = st.sidebar.selectbox(
        "Universe",
        group_options,
        index=group_options.index(universe_default),
        key="matrix_inspection::universe",
        format_func=_format_universe_label,
    )
    workspace_universe_group = universe_group
    start = _date_input_value("Start date", workspace_defaults.start, key="matrix_inspection::start_date")
    evaluation_start = _date_input_value("Evaluation start", workspace_defaults.evaluation_start, key="matrix_inspection::evaluation_start")
    evaluation_end = _date_input_value("Evaluation end", workspace_defaults.evaluation_end, key="matrix_inspection::evaluation_end")
    if st.sidebar.button("Refresh prices now"):
        _queue_force_refresh()
    if st.session_state.get(REFRESH_NEXT_RUN_KEY, False):
        st.sidebar.caption("Next run will force-refresh cached prices.")

workspace_context = build_workspace_context(
    config_path=config_path_input,
    config_defaults=config_defaults,
    universe_group=workspace_universe_group,
    universe=universe,
    start=start,
    evaluation_start=evaluation_start,
    evaluation_end=evaluation_end,
    refresh_pending=bool(st.session_state.get(REFRESH_NEXT_RUN_KEY, False)),
)

if usage_mode == "Workspace" and service_name == "Config editor":
    _render_config_editor(config_path_input, config_defaults)
elif usage_mode == "Inspection" and service_name == "Inspect at date":
    result_key = "matrix_inspection::snapshot::result"
    st.info("Inspect at date is the static matrix diagnostic view. Use it when you want one dated cleaned-matrix state with spectra, eigenvectors and cross-asset features.")
    cleaning_default = config_defaults.get("estimation", {}).get("cleaning_method", MATRIX_INSPECTION_CLEANING_OPTIONS[0])
    if cleaning_default == "rie":
        cleaning_default = "rie_spectral"
    shrinkage_default = float(config_defaults.get("estimation", {}).get("linear_shrinkage", 0.0) or 0.0)
    inspection_date_default = inspection_defaults.get("snapshot_date")
    matrix_type_default = str(inspection_defaults.get("snapshot_matrix_type", "correlation") or "correlation")
    input_type_default = str(inspection_defaults.get("snapshot_input_type", "normalized_returns") or "normalized_returns")
    estimator_method_default = str(inspection_defaults.get("snapshot_estimator_method", "sample_window") or "sample_window")
    window_default, num_assets = _universe_covariance_window_default(universe)
    st.caption(f"Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)")
    with st.form("matrix_inspection_snapshot_form"):
        row1 = st.columns(2)
        with row1[0]:
            matrix_type = st.selectbox(
                "Matrix type",
                MATRIX_TYPE_OPTIONS,
                index=MATRIX_TYPE_OPTIONS.index(matrix_type_default) if matrix_type_default in MATRIX_TYPE_OPTIONS else 0,
                format_func=lambda value: MATRIX_TYPE_LABELS.get(value, value),
            )
        with row1[1]:
            input_type = st.selectbox(
                "Input type",
                MATRIX_INPUT_OPTIONS,
                index=MATRIX_INPUT_OPTIONS.index(input_type_default) if input_type_default in MATRIX_INPUT_OPTIONS else 0,
                format_func=lambda value: MATRIX_INPUT_LABELS.get(value, value),
            )
        row2 = st.columns(3)
        with row2[0]:
            estimator_method = st.selectbox(
                "Estimator",
                MATRIX_ESTIMATOR_OPTIONS,
                index=MATRIX_ESTIMATOR_OPTIONS.index(estimator_method_default) if estimator_method_default in MATRIX_ESTIMATOR_OPTIONS else 0,
                format_func=lambda value: MATRIX_ESTIMATOR_LABELS.get(value, value),
            )
        with row2[1]:
            estimator_window = int(st.number_input("Estimator window", min_value=2, value=int(window_default), step=1))
        with row2[2]:
            st.caption("Derived EWMA alpha")
            estimator_alpha = alpha_from_span(estimator_window)
            st.code(f"{float(estimator_alpha):.6f}" if estimator_alpha is not None else "-", language="text")
        row3 = st.columns(2)
        with row3[0]:
            cleaning_method = st.selectbox(
                "Cleaning method",
                MATRIX_INSPECTION_CLEANING_OPTIONS,
                index=MATRIX_INSPECTION_CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in MATRIX_INSPECTION_CLEANING_OPTIONS else 0,
            )
        with row3[1]:
            linear_shrinkage_intensity = _linear_shrinkage_input(
                key="matrix_inspection::snapshot::linear_shrinkage",
                default_value=shrinkage_default,
            )
        row4 = st.columns(2)
        with row4[0]:
            use_latest = st.checkbox("Use latest available inspection date", value=inspection_date_default in (None, "", "None"))
        with row4[1]:
            inspection_date_selected = st.date_input("Inspection date", value=_parse_default_date(inspection_date_default).date(), disabled=use_latest)
        output_dir = st.text_input(
            "Output dir",
            value=str(inspection_defaults.get("snapshot_output_dir", "output/matrix_inspection/snapshot") or "output/matrix_inspection/snapshot"),
        )
        run_clicked = st.form_submit_button("Run inspect at date")
    if run_clicked:
        request = InspectionSnapshotRequest(
            refresh_policy=_consume_refresh_policy(),
            config_path=workspace_context.config_path,
            universe=workspace_context.universe,
            start=workspace_context.start or None,
            evaluation_start=workspace_context.evaluation_start or None,
            evaluation_end=workspace_context.evaluation_end or None,
            date=None if use_latest else pd.Timestamp(inspection_date_selected).date().isoformat(),
            cleaning_method=cleaning_method,
            input_type=input_type,
            matrix_type=matrix_type,
            estimator_method=estimator_method,
            linear_shrinkage_intensity=linear_shrinkage_intensity,
            estimator_window=int(estimator_window),
            output_dir=output_dir or None,
        )
        st.session_state[result_key] = run_inspection_snapshot(request)
    result = st.session_state.get(result_key)
    if result is not None:
        _render_snapshot_result(result, config_defaults=config_defaults)
elif usage_mode == "Inspection" and service_name == "Inspect over interval":
    result_key = "matrix_inspection::interval::result"
    st.info("Inspect over interval is the dynamic matrix diagnostic view. Use it when you want to study how spectra and leading eigenmodes evolve over a rebalance interval.")
    cleaning_default = config_defaults.get("estimation", {}).get("cleaning_method", MATRIX_INSPECTION_CLEANING_OPTIONS[0])
    if cleaning_default == "rie":
        cleaning_default = "rie_spectral"
    shrinkage_default = float(config_defaults.get("estimation", {}).get("linear_shrinkage", 0.0) or 0.0)
    freq_default = config_defaults.get("evaluation", {}).get("rebalance_frequency", FREQUENCY_OPTIONS[0])
    matrix_type_default = str(inspection_defaults.get("interval_matrix_type", "correlation") or "correlation")
    input_type_default = str(inspection_defaults.get("interval_input_type", "normalized_returns") or "normalized_returns")
    estimator_method_default = str(inspection_defaults.get("interval_estimator_method", "sample_window") or "sample_window")
    leading_default = int(inspection_defaults.get("leading_eigenvectors", 3) or 3)
    window_default, num_assets = _universe_covariance_window_default(universe)
    st.caption(f"Universe suggestion: {window_default} (1.5x {num_assets} assets, rounded up to the nearest 5)")
    with st.form("matrix_inspection_interval_form"):
        row1 = st.columns(2)
        with row1[0]:
            matrix_type = st.selectbox(
                "Matrix type",
                MATRIX_TYPE_OPTIONS,
                index=MATRIX_TYPE_OPTIONS.index(matrix_type_default) if matrix_type_default in MATRIX_TYPE_OPTIONS else 0,
                format_func=lambda value: MATRIX_TYPE_LABELS.get(value, value),
            )
        with row1[1]:
            input_type = st.selectbox(
                "Input type",
                MATRIX_INPUT_OPTIONS,
                index=MATRIX_INPUT_OPTIONS.index(input_type_default) if input_type_default in MATRIX_INPUT_OPTIONS else 0,
                format_func=lambda value: MATRIX_INPUT_LABELS.get(value, value),
            )
        row2 = st.columns(3)
        with row2[0]:
            estimator_method = st.selectbox(
                "Estimator",
                MATRIX_ESTIMATOR_OPTIONS,
                index=MATRIX_ESTIMATOR_OPTIONS.index(estimator_method_default) if estimator_method_default in MATRIX_ESTIMATOR_OPTIONS else 0,
                format_func=lambda value: MATRIX_ESTIMATOR_LABELS.get(value, value),
            )
        with row2[1]:
            estimator_window = int(st.number_input("Estimator window", min_value=2, value=int(window_default), step=1))
        with row2[2]:
            st.caption("Derived EWMA alpha")
            estimator_alpha = alpha_from_span(estimator_window)
            st.code(f"{float(estimator_alpha):.6f}" if estimator_alpha is not None else "-", language="text")
        row3 = st.columns(2)
        with row3[0]:
            cleaning_method = st.selectbox(
                "Cleaning method",
                MATRIX_INSPECTION_CLEANING_OPTIONS,
                index=MATRIX_INSPECTION_CLEANING_OPTIONS.index(cleaning_default) if cleaning_default in MATRIX_INSPECTION_CLEANING_OPTIONS else 0,
            )
        with row3[1]:
            linear_shrinkage_intensity = _linear_shrinkage_input(
                key="matrix_inspection::interval::linear_shrinkage",
                default_value=shrinkage_default,
            )
        row4 = st.columns(2)
        with row4[0]:
            rebalance_frequency = st.selectbox(
                "Inspection frequency",
                FREQUENCY_OPTIONS,
                index=FREQUENCY_OPTIONS.index(freq_default) if freq_default in FREQUENCY_OPTIONS else 0,
            )
        with row4[1]:
            leading_eigenvectors = int(st.number_input("Leading eigenvectors", min_value=1, max_value=12, value=leading_default, step=1))
        output_dir = st.text_input(
            "Output dir",
            value=str(inspection_defaults.get("interval_output_dir", "output/matrix_inspection/interval") or "output/matrix_inspection/interval"),
        )
        run_clicked = st.form_submit_button("Run inspect over interval")
    if run_clicked:
        request = InspectionIntervalRequest(
            refresh_policy=_consume_refresh_policy(),
            config_path=workspace_context.config_path,
            universe=workspace_context.universe,
            start=workspace_context.start or None,
            evaluation_start=workspace_context.evaluation_start or None,
            evaluation_end=workspace_context.evaluation_end or None,
            rebalance_frequency=rebalance_frequency,
            cleaning_method=cleaning_method,
            input_type=input_type,
            matrix_type=matrix_type,
            estimator_method=estimator_method,
            linear_shrinkage_intensity=linear_shrinkage_intensity,
            estimator_window=int(estimator_window),
            leading_eigenvectors=leading_eigenvectors,
            output_dir=output_dir or None,
        )
        st.session_state[result_key] = run_inspection_interval(request)
    result = st.session_state.get(result_key)
    if result is not None:
        _render_interval_result(result, config_defaults=config_defaults)
