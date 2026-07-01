from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import threading
from typing import Any, Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from market_tickers_data import MARKET_TICKERS
from optimal_tf.config_io import load_config
from optimal_tf.market_fork import MarketForkSnapshot, list_market_fork_snapshots, load_market_fork_snapshot
from optimal_tf.data import load_prices_yf
from optimal_tf.services import MarketSynthesisRequest, MarketSynthesisResult, run_market_synthesis

DEFAULT_CONFIG = "configs/optimal_tf.example.toml"
DEFAULT_OUTPUT_DIR = "output/optimal_tf/market_dashboard"
DEFAULT_FORK_DIR = "output/optimal_tf/market_forks"
GLOBAL_CACHE_WARM_START = "2000-01-01"
UNIVERSE_OPTIONS = sorted(MARKET_TICKERS)
MARKET_UNIVERSES = ["cac40", "dji", "eurostoxx50", "eurostoxx600", "nasdaq100", "sbf120", "sp500"]
INDEX_UNIVERSES = ["dataset_all", "futures", "index", "table8_all", "world_index", "test"]
UNIVERSE_GROUPS = {
    "Markets": [name for name in MARKET_UNIVERSES if name in MARKET_TICKERS],
    "Index universes": [name for name in INDEX_UNIVERSES if name in MARKET_TICKERS],
}
MOMENTUM_COLUMNS = ["annual", "semiannual", "quarterly", "monthly", "weekly", "daily"]
MOMENTUM_LABELS = {
    "annual": "1Y",
    "semiannual": "6M",
    "quarterly": "3M",
    "monthly": "1M",
    "weekly": "1W",
    "daily": "1D",
}
MARKET_SORT_OPTIONS = {
    "hierarchy": "Sector / sub-sector",
    "performance": "Performance",
}
MONTHLY_HISTORY_SORT_OPTIONS = {
    "hierarchy": "Sector / sub-sector",
    "last_month": "Last month",
    "trailing_3m": "Trailing 3M",
}
TABLE_HEADER_LABELS = {
    "sector": "Sector",
    "sub_sector": "Sub-\nsector",
    "category": "Category",
    "sub_category": "Sub-\ncategory",
    "ticker": "Ticker",
    "description": "Name",
    "num_tickers": "Num\nstocks",
    "name": "Artifact",
    "path": "Path",
}

st.set_page_config(page_title="market dashboard", layout="wide")
st.title("market dashboard")
st.caption("App autonome pour lancer la synthese marche et inspecter un fork issu d'optimal_tf.")

_CACHE_WARMER_LOCK = threading.Lock()
_CACHE_WARMER_STATE: dict[str, Any] = {
    "running": False,
    "total": 0,
    "done": 0,
    "failed": [],
    "current": None,
    "start": None,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Series):
        return value.rename("value").reset_index().to_dict(orient="records")
    if isinstance(value, pd.DataFrame):
        return value.head(200).to_dict(orient="records")
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
    return {
        "universe": asdict(universe),
        "estimation": asdict(estimation),
        "backtest": asdict(backtest),
        "allocation": asdict(allocation),
        "evaluation": asdict(evaluation),
        "compare": asdict(compare),
        "output": asdict(output),
    }, None


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


def _default_universe_group(universe: str) -> str:
    for group_name, options in UNIVERSE_GROUPS.items():
        if universe in options:
            return group_name
    return "Markets"


def _all_unique_base_tickers() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for tickers in MARKET_TICKERS.values():
        for ticker in tickers:
            if ticker not in seen:
                seen.add(ticker)
                ordered.append(ticker)
    return ordered


def _cache_warmer_snapshot() -> dict[str, Any]:
    with _CACHE_WARMER_LOCK:
        return {
            "running": bool(_CACHE_WARMER_STATE["running"]),
            "total": int(_CACHE_WARMER_STATE["total"]),
            "done": int(_CACHE_WARMER_STATE["done"]),
            "failed": list(_CACHE_WARMER_STATE["failed"]),
            "current": _CACHE_WARMER_STATE["current"],
            "start": _CACHE_WARMER_STATE["start"],
        }


def _run_cache_warmer(tickers: list[str], *, start: str) -> None:
    with _CACHE_WARMER_LOCK:
        _CACHE_WARMER_STATE.update(
            {
                "running": True,
                "total": len(tickers),
                "done": 0,
                "failed": [],
                "current": None,
                "start": start,
            }
        )
    try:
        for ticker in tickers:
            with _CACHE_WARMER_LOCK:
                _CACHE_WARMER_STATE["current"] = ticker
            try:
                load_prices_yf([ticker], start=start, refresh_policy="auto")
            except Exception as exc:  # pragma: no cover
                with _CACHE_WARMER_LOCK:
                    _CACHE_WARMER_STATE["failed"].append({"ticker": ticker, "error": str(exc)})
            finally:
                with _CACHE_WARMER_LOCK:
                    _CACHE_WARMER_STATE["done"] += 1
    finally:
        with _CACHE_WARMER_LOCK:
            _CACHE_WARMER_STATE["running"] = False
            _CACHE_WARMER_STATE["current"] = None


def _start_cache_warmer(*, start: str = GLOBAL_CACHE_WARM_START) -> bool:
    with _CACHE_WARMER_LOCK:
        if _CACHE_WARMER_STATE["running"]:
            return False
    tickers = _all_unique_base_tickers()
    worker = threading.Thread(
        target=_run_cache_warmer,
        kwargs={"tickers": tickers, "start": start},
        name="market-dashboard-cache-warmer",
        daemon=True,
    )
    worker.start()
    return True


def _render_cache_warmer_controls() -> None:
    st.sidebar.divider()
    st.sidebar.caption("Background cache warm-up")
    if st.sidebar.button("Warm all base tickers", key="market_app::warm_all_tickers"):
        started = _start_cache_warmer(start=GLOBAL_CACHE_WARM_START)
        if started:
            st.sidebar.success("Background warm-up started.")
        else:
            st.sidebar.info("Warm-up already running.")
    snapshot = _cache_warmer_snapshot()
    if snapshot["total"]:
        progress = 0.0 if snapshot["total"] <= 0 else snapshot["done"] / snapshot["total"]
        st.sidebar.progress(progress, text=f"{snapshot['done']} / {snapshot['total']} tickers")
        if snapshot["running"]:
            st.sidebar.caption(f"Current ticker: {snapshot['current'] or 'starting...'}")
            st.sidebar.caption(f"History start: {snapshot['start']}")
        else:
            st.sidebar.caption("Warm-up idle.")
        if snapshot["failed"]:
            st.sidebar.caption(f"Failures: {len(snapshot['failed'])}")
            preview = snapshot["failed"][:5]
            for item in preview:
                st.sidebar.caption(f"- {item['ticker']}")


def _maybe_autorefresh_cache_warmer(*, interval_ms: int = 3000) -> None:
    snapshot = _cache_warmer_snapshot()
    if not snapshot["running"]:
        return
    components.html(
        f"""
        <script>
        window.setTimeout(function() {{
            window.parent.location.reload();
        }}, {int(interval_ms)});
        </script>
        """,
        height=0,
        width=0,
    )


def _latest_or_date_input(label: str, default_value: Any, *, key_prefix: str, latest_default: bool = False) -> str | None:
    parsed = pd.Timestamp(default_value) if default_value not in (None, "", "None") else pd.Timestamp.today()
    use_latest = st.checkbox("Use latest available date", value=latest_default or default_value in (None, "", "None"), key=f"{key_prefix}::latest")
    selected = st.date_input(label, value=parsed.date(), disabled=use_latest, key=f"{key_prefix}::date")
    if use_latest:
        return None
    return pd.Timestamp(selected).date().isoformat()


def _compact_description(value: object, *, limit: int = 26) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rsplit(" ", 1)[0].strip()
    if not clipped:
        clipped = text[: limit - 1].strip()
    return f"{clipped}..."


def _compact_path(value: object, *, limit: int = 34) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    parts = text.split("/")
    if len(parts) >= 2:
        candidate = f".../{parts[-2]}/{parts[-1]}"
        if len(candidate) <= limit:
            return candidate
    return f"...{text[-(limit - 3):]}"


def _prepare_market_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    if "description" in table.columns:
        table["description"] = table["description"].map(_compact_description)
    if "path" in table.columns:
        table["path"] = table["path"].map(_compact_path)
    return table


def _merge_repeated_labels(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty or not columns:
        return frame
    merged = frame.copy()
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
                merged.at[idx, column] = ""
                reset_lower = False
            else:
                previous_values[column] = value
                reset_lower = True
    return merged


def _rename_momentum_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: f"{MOMENTUM_LABELS[column]}\n(%)" for column in frame.columns if column in MOMENTUM_LABELS}
    return frame.rename(columns=rename_map)


def _rename_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: TABLE_HEADER_LABELS[column] for column in frame.columns if column in TABLE_HEADER_LABELS}
    return frame.rename(columns=rename_map)


def _render_momentum_table(
    frame: pd.DataFrame,
    *,
    empty_message: str = "No momentum data available.",
    merge_columns: Iterable[str] = (),
) -> None:
    if frame.empty:
        st.info(empty_message)
        return
    table = _prepare_market_display_columns(frame.copy())
    table = _merge_repeated_labels(table, [column for column in merge_columns if column in table.columns])
    for column in MOMENTUM_COLUMNS:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce") * 100.0
    table = _rename_display_columns(_rename_momentum_columns(table))
    momentum_cols = [f"{MOMENTUM_LABELS[column]}\n(%)" for column in MOMENTUM_COLUMNS if f"{MOMENTUM_LABELS[column]}\n(%)" in table.columns]
    bound = float(np.nanmax(np.abs(table[momentum_cols].to_numpy(dtype=float)))) if momentum_cols else 0.0
    styled = (
        table.style
        .format({column: "{:.1f}" for column in momentum_cols})
        .background_gradient(cmap="RdYlGn", subset=momentum_cols, vmin=-max(bound, 1e-6), vmax=max(bound, 1e-6), axis=None)
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "8px"), ("padding", "1px 4px"), ("line-height", "0.9")]},
            {"selector": "td", "props": [("font-size", "8px"), ("padding", "1px 4px"), ("line-height", "0.9")]},
        ])
    )
    st.table(styled)


def _sort_market_frame(frame: pd.DataFrame, sort_by: str, *, sort_mode: str = "performance") -> pd.DataFrame:
    if frame.empty:
        return frame
    if sort_mode == "hierarchy":
        hierarchy_cols = [column for column in ["sector", "sub_sector", "category", "ticker"] if column in frame.columns]
        return frame.sort_values(hierarchy_cols, kind="stable").reset_index(drop=True) if hierarchy_cols else frame.reset_index(drop=True)
    if sort_by not in frame.columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_by, ascending=False, kind="stable").reset_index(drop=True)


def _render_market_overview(
    result: MarketSynthesisResult,
    *,
    ranking_column: str,
    ranking_label: str,
    top_n: int,
) -> None:
    ticker_detail = result.ticker_frame.reset_index() if isinstance(result.ticker_frame.index, pd.MultiIndex) else result.ticker_frame.reset_index().rename(columns={"index": "ticker"})
    ranked = _sort_market_frame(ticker_detail, ranking_column, sort_mode="performance")
    positive_share = float((ranked[ranking_column] > 0).mean()) if not ranked.empty else 0.0
    negative_share = float((ranked[ranking_column] < 0).mean()) if not ranked.empty else 0.0
    metric_cols = st.columns(4)
    metric_cols[0].metric("As of date", result.as_of_date.strftime("%Y-%m-%d"))
    metric_cols[1].metric("Universe", result.universe.upper())
    metric_cols[2].metric(f"Positive {ranking_label}", f"{positive_share:.0%}")
    metric_cols[3].metric(f"Negative {ranking_label}", f"{negative_share:.0%}")
    if not ranked.empty:
        leader_cols = st.columns(2)
        best = ranked.iloc[0]
        worst = ranked.iloc[-1]
        leader_cols[0].metric("Best ticker", str(best.get("ticker", best.name)), delta=f"{best[ranking_column]:.2%}")
        leader_cols[1].metric("Weakest ticker", str(worst.get("ticker", worst.name)), delta=f"{worst[ranking_column]:.2%}")
    cols = [column for column in ["sector", "sub_sector", "category", "ticker", "description", *MOMENTUM_COLUMNS] if column in ranked.columns]
    st.caption(f"Top {top_n} tickers on {ranking_label}")
    _render_momentum_table(ranked[cols].head(top_n))
    st.caption(f"Bottom {top_n} tickers on {ranking_label}")
    _render_momentum_table(ranked[cols].tail(top_n).sort_values(ranking_column, ascending=True, kind="stable").reset_index(drop=True))


def _market_nav_chart_frame(frame: pd.DataFrame, *, lookback: str, sampling: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    periods = {"1Y": 252, "2Y": 504}.get(lookback, 252)
    window = frame.sort_index().tail(min(periods, len(frame))).copy()
    if window.empty:
        return window
    window = window.ffill().dropna(how="all")
    if window.empty:
        return window
    rebased = 100.0 * window.divide(window.iloc[0].replace(0.0, np.nan))
    rebased = rebased.dropna(how="all")
    if rebased.empty:
        return rebased
    if sampling == "Weekly":
        rebased = rebased.resample("W-FRI").last().dropna(how="all")
    return rebased


def _market_price_chart_frame(frame: pd.DataFrame, *, lookback: str, sampling: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    periods = {"1Y": 252, "2Y": 504}.get(lookback, 252)
    window = frame.sort_index().tail(min(periods, len(frame))).copy()
    if window.empty:
        return window
    window = window.ffill().dropna(how="all")
    if window.empty:
        return window
    if sampling == "Weekly":
        window = window.resample("W-FRI").last().dropna(how="all")
    return window


def _ticker_detail_frame(result: MarketSynthesisResult) -> pd.DataFrame:
    if isinstance(result.ticker_frame.index, pd.MultiIndex):
        return result.ticker_frame.reset_index()
    return result.ticker_frame.reset_index().rename(columns={"index": "ticker"})


def _ticker_history_frame(result: MarketSynthesisResult) -> pd.DataFrame:
    if isinstance(result.monthly_ticker_frame.index, pd.MultiIndex):
        return result.monthly_ticker_frame.reset_index()
    return result.monthly_ticker_frame.reset_index().rename(columns={"index": "ticker"})


def _select_filter_options(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    values = sorted({str(value).strip() for value in frame[column].dropna().tolist() if str(value).strip()})
    return values


def _fallback_ticker_nav_frame(result: MarketSynthesisResult, ticker: str) -> pd.DataFrame:
    try:
        prices = load_prices_yf([ticker], start=result.start, refresh_policy=result.request.refresh_policy)
    except Exception:
        return pd.DataFrame()
    if ticker not in prices.columns:
        return pd.DataFrame()
    history = prices.loc[prices.index <= result.as_of_date, [ticker]].ffill().dropna(how="all")
    if history.empty:
        return pd.DataFrame()
    first_valid = history[ticker].dropna()
    if first_valid.empty or float(first_valid.iloc[0]) == 0.0:
        return pd.DataFrame()
    return 100.0 * history.divide(float(first_valid.iloc[0]))


def _load_ticker_nav_series(
    ticker: str,
    start: str,
    as_of_date: str,
    refresh_policy: str,
) -> pd.DataFrame:
    try:
        # Reuse the existing local cache first; this avoids fragile per-ticker
        # live refreshes for some European symbols when the market synthesis
        # already loaded valid data earlier in the session.
        prices = load_prices_yf([ticker], start=start, refresh_policy="never")
    except Exception:
        try:
            prices = load_prices_yf([ticker], start=start, refresh_policy=refresh_policy)
        except Exception:
            return pd.DataFrame()
    if ticker not in prices.columns:
        return pd.DataFrame()
    history = prices.loc[prices.index <= pd.Timestamp(as_of_date), [ticker]].ffill().dropna(how="all")
    return history


def _full_history_price_chart_frame(frame: pd.DataFrame, *, sampling: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    chart = frame.sort_index().ffill().dropna(how="all")
    if chart.empty:
        return chart
    if sampling == "Weekly":
        chart = chart.resample("W-FRI").last().dropna(how="all")
    return chart


def _full_history_nav_chart_frame(frame: pd.DataFrame, *, sampling: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    chart = frame.sort_index().ffill().dropna(how="all")
    if chart.empty:
        return chart
    rebased = 100.0 * chart.divide(chart.iloc[0].replace(0.0, np.nan))
    rebased = rebased.dropna(how="all")
    if sampling == "Weekly":
        rebased = rebased.resample("W-FRI").last().dropna(how="all")
    return rebased


def _render_tinted_timeseries_chart(
    frame: pd.DataFrame,
    *,
    value_label: str,
    title: str,
    area_color: str,
    line_color: str,
) -> None:
    if frame.empty:
        return
    series_name = str(frame.columns[0])
    plot = frame.rename(columns={series_name: value_label}).reset_index()
    x_column = plot.columns[0]
    values = pd.to_numeric(plot[value_label], errors="coerce").dropna()
    if values.empty:
        return
    ymin = float(values.min())
    ymax = float(values.max())
    spread = ymax - ymin
    padding = max(spread * 0.06, abs(ymax) * 0.01, 1e-6)
    y_lower = ymin - padding
    y_upper = ymax + padding
    plot["baseline"] = y_lower
    y_scale = alt.Scale(domain=[y_lower, y_upper], zero=False)
    area = (
        alt.Chart(plot)
        .mark_area(color=area_color, opacity=0.28, clip=True)
        .encode(
            x=alt.X(f"{x_column}:T", title=None),
            y=alt.Y(f"{value_label}:Q", title=value_label, scale=y_scale),
            y2=alt.Y2("baseline:Q"),
        )
    )
    line = (
        alt.Chart(plot)
        .mark_line(color=line_color, strokeWidth=2.2, clip=True)
        .encode(
            x=alt.X(f"{x_column}:T", title=None),
            y=alt.Y(f"{value_label}:Q", title=value_label, scale=y_scale),
            tooltip=[
                alt.Tooltip(f"{x_column}:T", title="Date"),
                alt.Tooltip(f"{value_label}:Q", title=value_label, format=".2f"),
            ],
        )
    )
    chart = (area + line).properties(title=title, height=260).configure(
        axis=alt.AxisConfig(gridColor="#d7dbe2", labelColor="#2f3540", titleColor="#2f3540"),
        view=alt.ViewConfig(strokeOpacity=0),
    )
    st.altair_chart(chart, width="stretch")


def _render_ticker_view(result: MarketSynthesisResult, *, key_prefix: str) -> None:
    detail_frame = _ticker_detail_frame(result)
    if detail_frame.empty or "ticker" not in detail_frame.columns:
        st.info("No ticker data available.")
        return

    filtered = detail_frame.copy()
    has_sector_path = bool(_select_filter_options(detail_frame, "sector"))
    has_category_path = bool(_select_filter_options(detail_frame, "category"))
    selector_mode_options = []
    if has_sector_path:
        selector_mode_options.append("Sector / Sub-sector")
    if has_category_path:
        selector_mode_options.append("Category / Sub-category")
    if not selector_mode_options:
        selector_mode_options.append("All tickers")

    selector_mode = st.radio(
        "Filter mode",
        selector_mode_options,
        horizontal=True,
        key=f"{key_prefix}::filter_mode",
    )

    if selector_mode == "Sector / Sub-sector":
        selector_cols = st.columns(2)
        sector_options = _select_filter_options(detail_frame, "sector")
        with selector_cols[0]:
            selected_sector = st.selectbox(
                "Sector",
                ["All", *sector_options] if sector_options else ["All"],
                key=f"{key_prefix}::sector",
            )
        if selected_sector != "All" and "sector" in filtered.columns:
            filtered = filtered.loc[filtered["sector"] == selected_sector].copy()

        sub_sector_options = _select_filter_options(filtered if "sub_sector" in filtered.columns else detail_frame, "sub_sector")
        with selector_cols[1]:
            selected_sub_sector = st.selectbox(
                "Sub-sector",
                ["All", *sub_sector_options] if sub_sector_options else ["All"],
                key=f"{key_prefix}::sub_sector",
            )
        if selected_sub_sector != "All" and "sub_sector" in filtered.columns:
            filtered = filtered.loc[filtered["sub_sector"] == selected_sub_sector].copy()
    elif selector_mode == "Category / Sub-category":
        selector_cols = st.columns(2)
        category_options = _select_filter_options(detail_frame, "category")
        with selector_cols[0]:
            selected_category = st.selectbox(
                "Category",
                ["All", *category_options] if category_options else ["All"],
                key=f"{key_prefix}::category",
            )
        if selected_category != "All" and "category" in filtered.columns:
            filtered = filtered.loc[filtered["category"] == selected_category].copy()

        sub_category_options = _select_filter_options(filtered if "sub_category" in filtered.columns else detail_frame, "sub_category")
        with selector_cols[1]:
            selected_sub_category = st.selectbox(
                "Sub-category",
                ["All", *sub_category_options] if sub_category_options else ["All"],
                key=f"{key_prefix}::sub_category",
            )
        if selected_sub_category != "All" and "sub_category" in filtered.columns:
            filtered = filtered.loc[filtered["sub_category"] == selected_sub_category].copy()

    if filtered.empty:
        st.info("No ticker matches the selected filters.")
        return

    options = filtered[["ticker", "description"]].drop_duplicates().copy()
    options["label"] = options.apply(
        lambda row: str(row["ticker"]) if not str(row["description"]).strip() else f"{row['ticker']} - {row['description']}",
        axis=1,
    )
    option_labels = options["label"].tolist()
    selected_label = st.selectbox("Ticker", option_labels, key=f"{key_prefix}::ticker")
    selected_ticker = str(options.loc[options["label"] == selected_label, "ticker"].iloc[0])
    selected_row = filtered.loc[filtered["ticker"] == selected_ticker].iloc[0]

    info_cols = st.columns(4)
    info_cols[0].metric("Ticker", selected_ticker)
    info_cols[1].metric("Category", str(selected_row.get("category", "")) or "-")
    info_cols[2].metric("Sector", str(selected_row.get("sector", "")) or "-")
    info_cols[3].metric("Sub-sector", str(selected_row.get("sub_sector", "")) or "-")
    if str(selected_row.get("description", "")).strip():
        st.caption(str(selected_row["description"]))

    nav_cols = st.columns(2)
    with nav_cols[0]:
        nav_lookback = st.selectbox("Lookback", ["1Y", "2Y"], index=0, key=f"{key_prefix}::nav_lookback")
    with nav_cols[1]:
        nav_sampling = st.selectbox("Sampling", ["Daily", "Weekly"], index=0, key=f"{key_prefix}::nav_sampling")

    ticker_nav_frame = _load_ticker_nav_series(
        selected_ticker,
        result.start,
        result.as_of_date.strftime("%Y-%m-%d"),
        result.request.refresh_policy,
    )
    price_frame = _market_price_chart_frame(ticker_nav_frame, lookback=nav_lookback, sampling=nav_sampling)
    price_points = int(price_frame.dropna(how="all").shape[0]) if not price_frame.empty else 0
    if price_points == 0:
        price_frame = _full_history_price_chart_frame(ticker_nav_frame, sampling=nav_sampling)
        price_points = int(price_frame.dropna(how="all").shape[0]) if not price_frame.empty else 0
    chart_frame = _market_nav_chart_frame(ticker_nav_frame, lookback=nav_lookback, sampling=nav_sampling)
    chart_points = int(chart_frame.dropna(how="all").shape[0]) if not chart_frame.empty else 0
    if chart_points == 0:
        chart_frame = _full_history_nav_chart_frame(ticker_nav_frame, sampling=nav_sampling)
        chart_points = int(chart_frame.dropna(how="all").shape[0]) if not chart_frame.empty else 0
        if chart_points > 0:
            st.caption("Selected lookback has no points; showing full available history instead.")
    if chart_points == 0:
        st.info("No NAV points are available for this ticker on the selected lookback.")
        return
    if price_points > 0:
        _render_tinted_timeseries_chart(
            price_frame.rename(columns={selected_ticker: "Price"}),
            value_label="Price",
            title="Ticker price",
            area_color="#d89b2b",
            line_color="#9a5a00",
        )
    nav_display = chart_frame.rename(columns={selected_ticker: "NAV"}).copy()
    nav_display["NAV"] = pd.to_numeric(nav_display["NAV"], errors="coerce") - 100.0
    st.caption("Ticker NAV rebased to 100.")
    _render_tinted_timeseries_chart(
        nav_display,
        value_label="NAV",
        title="Ticker NAV rebased to 0",
        area_color="#2f8f6b",
        line_color="#165a43",
    )


def _monthly_history_columns(frame: pd.DataFrame) -> list[str]:
    metadata_cols = {"level", "sector", "sub_sector", "category", "sub_category", "description", "label", "num_tickers", "hierarchy_complete", "category_complete", "ticker"}
    return [column for column in frame.columns if column not in metadata_cols]


def _prepare_monthly_history_frame(frame: pd.DataFrame, *, lookback_months: int, sort_mode: str) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty:
        return frame, []
    month_columns = _monthly_history_columns(frame)
    if not month_columns:
        return frame, []
    selected_months = list(reversed(month_columns[-lookback_months:]))
    metadata_columns = [column for column in frame.columns if column not in month_columns]
    table = frame.loc[:, metadata_columns + selected_months].copy()
    if sort_mode == "hierarchy":
        table = _sort_market_frame(table, selected_months[0], sort_mode="hierarchy")
    elif sort_mode == "last_month":
        table = table.sort_values(selected_months[0], ascending=False, kind="stable").reset_index(drop=True)
    else:
        trailing_cols = selected_months[: min(3, len(selected_months))]
        table = table.assign(_trailing_3m=table[trailing_cols].mean(axis=1))
        table = table.sort_values("_trailing_3m", ascending=False, kind="stable").drop(columns=["_trailing_3m"]).reset_index(drop=True)
    return table, selected_months


def _render_monthly_history_table(
    frame: pd.DataFrame,
    *,
    month_columns: list[str],
    empty_message: str = "No monthly history available.",
    merge_columns: Iterable[str] = (),
    font_px: int = 9,
) -> None:
    if frame.empty or not month_columns:
        st.info(empty_message)
        return
    table = _merge_repeated_labels(_prepare_market_display_columns(frame.copy()), [column for column in merge_columns if column in frame.columns])
    rename_map = {column: f"{column}\n(%)" for column in month_columns}
    for column in month_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce") * 100.0
    table = _rename_display_columns(table).rename(columns=rename_map)
    month_display_columns = [rename_map[column] for column in month_columns]
    bound = float(np.nanmax(np.abs(table[month_display_columns].to_numpy(dtype=float)))) if month_display_columns else 0.0
    styled = (
        table.style
        .format({column: "{:.1f}" for column in month_display_columns})
        .background_gradient(cmap="RdYlGn", subset=month_display_columns, vmin=-max(bound, 1e-6), vmax=max(bound, 1e-6), axis=None)
        .set_table_styles([
            {"selector": "th", "props": [("font-size", f"{font_px}px"), ("padding", "1px 4px"), ("line-height", "0.9")]},
            {"selector": "td", "props": [("font-size", f"{font_px}px"), ("padding", "1px 4px"), ("line-height", "0.9")]},
        ])
    )
    st.table(styled)


def _render_artifacts_block(files: dict[str, str]) -> None:
    if not files:
        st.info("No artifacts available.")
        return
    frame = pd.DataFrame([{"name": name, "path": path} for name, path in files.items()])
    st.dataframe(frame, width="stretch", hide_index=True)


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


def _service_result_view(options: list[str], *, key: str) -> str:
    return st.radio(
        "View",
        options,
        key=key,
        horizontal=True,
        label_visibility="collapsed",
    )


def _snapshot_choice_label(path: Path) -> str:
    return path.name


def _render_snapshot_block(snapshot: MarketForkSnapshot) -> None:
    st.subheader("Fork snapshot")
    meta = {
        "label": snapshot.label,
        "created_at_utc": snapshot.created_at_utc,
        "source_service": snapshot.source_service,
        "config_path": snapshot.config_path,
        "market_universe": snapshot.market_universe,
        "market_start": snapshot.market_start,
        "market_as_of_date": snapshot.market_as_of_date,
    }
    st.json(meta, expanded=False)
    if snapshot.source_context:
        st.caption("Source context")
        st.json(_json_safe(snapshot.source_context), expanded=False)
    if snapshot.source_request:
        st.caption("Source request")
        st.json(_json_safe(snapshot.source_request), expanded=False)
    st.caption("Source artifacts")
    _render_artifacts_block(snapshot.source_artifacts)


snapshot_path_from_query = str(st.query_params.get("fork", ""))
default_snapshot_dir = str(Path(snapshot_path_from_query).parent) if snapshot_path_from_query else DEFAULT_FORK_DIR
snapshot_dir = st.sidebar.text_input("Fork snapshot dir", value=default_snapshot_dir)
recent_snapshot_paths = list_market_fork_snapshots(snapshot_dir)
recent_snapshot_options = [""] + [str(path) for path in recent_snapshot_paths]
default_recent_snapshot = snapshot_path_from_query if snapshot_path_from_query in recent_snapshot_options else (recent_snapshot_options[1] if len(recent_snapshot_options) > 1 else "")
selected_recent_snapshot = st.sidebar.selectbox(
    "Recent fork snapshots",
    recent_snapshot_options,
    index=recent_snapshot_options.index(default_recent_snapshot) if default_recent_snapshot in recent_snapshot_options else 0,
    format_func=lambda value: "None" if not value else _snapshot_choice_label(Path(value)),
)
snapshot_path_default = selected_recent_snapshot or snapshot_path_from_query
snapshot_path = st.sidebar.text_input(
    "Fork snapshot path",
    value=snapshot_path_default,
    help="Use a recent snapshot above, or provide a custom JSON path here.",
)
snapshot: MarketForkSnapshot | None = None
snapshot_error: str | None = None
if snapshot_path:
    try:
        snapshot = load_market_fork_snapshot(snapshot_path)
    except Exception as exc:  # pragma: no cover
        snapshot_error = str(exc)

config_path_default = snapshot.config_path if snapshot else DEFAULT_CONFIG
config_path = st.sidebar.text_input("Config path", value=config_path_default)
config_defaults, config_error = _load_defaults(config_path)
if config_error:
    st.sidebar.error(f"Config load failed: {config_error}")
    config_defaults = {}
if snapshot_error:
    st.sidebar.error(f"Snapshot load failed: {snapshot_error}")
elif snapshot_path:
    st.sidebar.caption(f"Loaded snapshot: {Path(snapshot_path).name}")

_render_cache_warmer_controls()
_maybe_autorefresh_cache_warmer()

universe_default = snapshot.market_universe if snapshot else config_defaults.get("universe", {}).get("name", UNIVERSE_OPTIONS[0])
start_default = snapshot.market_start if snapshot else config_defaults.get("universe", {}).get("start", "")
date_default = snapshot.market_as_of_date if snapshot else config_defaults.get("evaluation", {}).get("evaluation_end")

group_default = _default_universe_group(universe_default)
group_names = [name for name, options in UNIVERSE_GROUPS.items() if options]
group_index = group_names.index(group_default) if group_default in group_names else 0
universe_group = st.sidebar.selectbox("Universe group", group_names, index=group_index)
group_options = UNIVERSE_GROUPS.get(universe_group, UNIVERSE_OPTIONS)
if not group_options:
    group_options = UNIVERSE_OPTIONS
universe_index = group_options.index(universe_default) if universe_default in group_options else 0
universe = st.sidebar.selectbox("Universe", group_options, index=universe_index, format_func=_format_universe_label)
start = st.sidebar.text_input("Start", value=start_default or "")
market_date = _latest_or_date_input("Market date", date_default, key_prefix="market_app::date", latest_default=not bool(snapshot and snapshot.market_as_of_date))
output_dir = st.sidebar.text_input("Output dir", value=DEFAULT_OUTPUT_DIR)

if st.sidebar.button("Run market synthesis"):
    request = MarketSynthesisRequest(
        config_path=config_path,
        universe=universe,
        start=start or None,
        as_of_date=market_date or None,
        output_dir=output_dir or None,
    )
    st.session_state["market_app::result"] = run_market_synthesis(request)

result = st.session_state.get("market_app::result")
section_options = ["Results", "Config", "Artifacts"]
if snapshot is not None:
    section_options.append("Fork")
section = _service_result_view(section_options, key="market_app::view")

if section == "Fork" and snapshot is not None:
    _render_snapshot_block(snapshot)
elif result is None:
    st.info("Run a market synthesis to populate this app.")
    if snapshot is not None:
        _render_snapshot_block(snapshot)
else:
    if section == "Results":
        if result.has_hierarchy:
            overview_tab, sector_tab, sub_sector_tab, tickers_tab, ticker_tab = st.tabs(["Overview", "Sector", "Sub-sector", "Tickers", "Ticker"])
            with overview_tab:
                ranking_col, topn_col = st.columns(2)
                with ranking_col:
                    overview_ranking = st.selectbox(
                        "Ranking horizon",
                        MOMENTUM_COLUMNS,
                        format_func=lambda value: MOMENTUM_LABELS[value],
                        index=0,
                        key="market_app::overview_ranking",
                    )
                with topn_col:
                    overview_top_n = int(st.slider("Top / bottom rows", min_value=3, max_value=12, value=6, step=1, key="market_app::overview_top_n"))
                st.subheader("Market synthesis")
                st.caption(f"Quick read of the market ranked by {MOMENTUM_LABELS[overview_ranking].lower()} momentum.")
                _render_market_overview(result, ranking_column=overview_ranking, ranking_label=MOMENTUM_LABELS[overview_ranking], top_n=overview_top_n)
            with sector_tab:
                momentum_tab, history_tab, nav_tab = st.tabs(["Momentum", "Monthly history", "NAV"])
                with momentum_tab:
                    left, right = st.columns(2)
                    with left:
                        sector_ranking = st.selectbox("Ranking horizon", MOMENTUM_COLUMNS, format_func=lambda value: MOMENTUM_LABELS[value], index=0, key="market_app::sector_ranking")
                    with right:
                        sector_sort = st.selectbox("Sort mode", list(MARKET_SORT_OPTIONS), format_func=lambda value: MARKET_SORT_OPTIONS[value], index=0, key="market_app::sector_sort")
                    sector_frame = _sort_market_frame(
                        result.consolidated_frame.loc[result.consolidated_frame["level"] == "sector", ["sector", "num_tickers", *MOMENTUM_COLUMNS]].copy(),
                        sector_ranking,
                        sort_mode="hierarchy" if sector_sort == "hierarchy" else "performance",
                    )
                    _render_momentum_table(sector_frame, empty_message="No sector momentum available.")
                with history_tab:
                    left, right = st.columns(2)
                    with left:
                        lookback_months = int(st.selectbox("Lookback months", [6, 12, 18, 24], index=1, key="market_app::sector_months"))
                    with right:
                        history_sort = st.selectbox("Sort mode", list(MONTHLY_HISTORY_SORT_OPTIONS), format_func=lambda value: MONTHLY_HISTORY_SORT_OPTIONS[value], index=0, key="market_app::sector_history_sort")
                    monthly_frame, month_columns = _prepare_monthly_history_frame(result.monthly_consolidated_frame.loc[result.monthly_consolidated_frame["level"] == "sector"].copy(), lookback_months=lookback_months, sort_mode=history_sort)
                    monthly_frame = monthly_frame.loc[:, ["sector", *month_columns]]
                    st.caption(f"Last {lookback_months} monthly returns by sector.")
                    _render_monthly_history_table(monthly_frame, month_columns=month_columns)
                with nav_tab:
                    left, right = st.columns(2)
                    with left:
                        nav_lookback = st.selectbox("Lookback", ["1Y", "2Y"], index=0, key="market_app::sector_nav_lookback")
                    with right:
                        nav_sampling = st.selectbox("Sampling", ["Daily", "Weekly"], index=0, key="market_app::sector_nav_sampling")
                    st.caption("Equal-weight sector NAV rebased to 100.")
                    st.line_chart(_market_nav_chart_frame(result.sector_nav_frame, lookback=nav_lookback, sampling=nav_sampling))
            with sub_sector_tab:
                sub_momentum, sub_history = st.tabs(["Momentum", "Monthly history"])
                with sub_momentum:
                    left, right = st.columns(2)
                    with left:
                        sub_ranking = st.selectbox("Ranking horizon", MOMENTUM_COLUMNS, format_func=lambda value: MOMENTUM_LABELS[value], index=0, key="market_app::sub_ranking")
                    with right:
                        sub_sort = st.selectbox("Sort mode", list(MARKET_SORT_OPTIONS), format_func=lambda value: MARKET_SORT_OPTIONS[value], index=0, key="market_app::sub_sort")
                    sub_frame = _sort_market_frame(
                        result.consolidated_frame.loc[result.consolidated_frame["level"] == "sub_sector", ["sector", "sub_sector", "num_tickers", *MOMENTUM_COLUMNS]].copy(),
                        sub_ranking,
                        sort_mode="hierarchy" if sub_sort == "hierarchy" else "performance",
                    )
                    _render_momentum_table(sub_frame, empty_message="No sub-sector momentum available.", merge_columns=["sector"])
                with sub_history:
                    left, right = st.columns(2)
                    with left:
                        lookback_months = int(st.selectbox("Lookback months", [6, 12, 18, 24], index=1, key="market_app::sub_months"))
                    with right:
                        history_sort = st.selectbox("Sort mode", list(MONTHLY_HISTORY_SORT_OPTIONS), format_func=lambda value: MONTHLY_HISTORY_SORT_OPTIONS[value], index=0, key="market_app::sub_history_sort")
                    monthly_frame, month_columns = _prepare_monthly_history_frame(result.monthly_consolidated_frame.loc[result.monthly_consolidated_frame["level"] == "sub_sector"].copy(), lookback_months=lookback_months, sort_mode=history_sort)
                    monthly_frame = monthly_frame.loc[:, ["sector", "sub_sector", *month_columns]]
                    st.caption(f"Last {lookback_months} monthly returns by sub-sector.")
                    _render_monthly_history_table(monthly_frame, month_columns=month_columns, merge_columns=["sector"])
            with tickers_tab:
                detail_momentum, detail_history = st.tabs(["Momentum", "Monthly history"])
                detail_frame = _ticker_detail_frame(result)
                detail_history_frame = _ticker_history_frame(result)
                with detail_momentum:
                    left, right = st.columns(2)
                    with left:
                        ranking = st.selectbox("Ranking horizon", MOMENTUM_COLUMNS, format_func=lambda value: MOMENTUM_LABELS[value], index=0, key="market_app::detail_ranking")
                    with right:
                        detail_sort = st.selectbox("Sort mode", list(MARKET_SORT_OPTIONS), format_func=lambda value: MARKET_SORT_OPTIONS[value], index=0, key="market_app::detail_sort")
                    sorted_detail = _sort_market_frame(detail_frame, ranking, sort_mode="hierarchy" if detail_sort == "hierarchy" else "performance")
                    sectors = sorted_detail["sector"].drop_duplicates().tolist()
                    selected_sector = st.selectbox("Sector", sectors, key="market_app::detail_sector")
                    sector_frame = sorted_detail.loc[sorted_detail["sector"] == selected_sector].copy()
                    _render_momentum_table(sector_frame[["sub_sector", "ticker", "description", *MOMENTUM_COLUMNS]], empty_message="No ticker detail available.", merge_columns=["sub_sector"])
                with detail_history:
                    left, center, right = st.columns(3)
                    with left:
                        lookback_months = int(st.selectbox("Lookback months", [6, 12, 18, 24], index=1, key="market_app::detail_months"))
                    with center:
                        detail_history_sort = st.selectbox("Sort mode", ["last_month", "trailing_3m"], format_func=lambda value: {"last_month": "Last month", "trailing_3m": "Trailing 3M"}[value], index=0, key="market_app::detail_history_sort")
                    sectors = detail_history_frame["sector"].drop_duplicates().tolist()
                    with right:
                        selected_sector_history = st.selectbox("Sector", sectors, key="market_app::detail_sector_history")
                    filtered = detail_history_frame.loc[
                        detail_history_frame["sector"] == selected_sector_history,
                        ["sub_sector", "ticker", "description", *_monthly_history_columns(detail_history_frame)],
                    ].reset_index(drop=True)
                    prepared, month_columns = _prepare_monthly_history_frame(filtered, lookback_months=lookback_months, sort_mode=detail_history_sort)
                    _render_monthly_history_table(prepared, month_columns=month_columns, merge_columns=["sub_sector"])
            with ticker_tab:
                _render_ticker_view(result, key_prefix="market_app::hierarchy_ticker")
        elif result.synthesis_mode == "category":
            overview_tab, category_tab, tickers_tab, ticker_tab = st.tabs(["Overview", "Category", "Tickers", "Ticker"])
            with overview_tab:
                ranking_col, topn_col = st.columns(2)
                with ranking_col:
                    overview_ranking = st.selectbox(
                        "Ranking horizon",
                        MOMENTUM_COLUMNS,
                        format_func=lambda value: MOMENTUM_LABELS[value],
                        index=0,
                        key="market_app::overview_ranking",
                    )
                with topn_col:
                    overview_top_n = int(st.slider("Top / bottom rows", min_value=3, max_value=12, value=6, step=1, key="market_app::overview_top_n"))
                st.subheader("Market synthesis")
                st.caption(f"Quick read of the market ranked by {MOMENTUM_LABELS[overview_ranking].lower()} momentum.")
                _render_market_overview(result, ranking_column=overview_ranking, ranking_label=MOMENTUM_LABELS[overview_ranking], top_n=overview_top_n)
            with category_tab:
                category_momentum_tab, category_history_tab = st.tabs(["Momentum", "Monthly history"])
                with category_momentum_tab:
                    detail_frame = result.ticker_frame.reset_index() if isinstance(result.ticker_frame.index, pd.MultiIndex) else result.ticker_frame.reset_index().rename(columns={"index": "ticker"})
                    left, right = st.columns(2)
                    with left:
                        category_ranking = st.selectbox("Ranking horizon", MOMENTUM_COLUMNS, format_func=lambda value: MOMENTUM_LABELS[value], index=0, key="market_app::category_ranking")
                    with right:
                        category_sort = st.selectbox("Sort mode", ["hierarchy", "performance"], format_func=lambda value: {"hierarchy": "Category", "performance": "Performance"}[value], index=0, key="market_app::category_sort")
                    detail_frame = _sort_market_frame(detail_frame, category_ranking, sort_mode="hierarchy" if category_sort == "hierarchy" else "performance")
                    categories = detail_frame["category"].drop_duplicates().tolist() if "category" in detail_frame.columns else []
                    for category in categories:
                        with st.expander(str(category), expanded=False):
                            category_frame = detail_frame.loc[detail_frame["category"] == category, ["ticker", "description", *MOMENTUM_COLUMNS]].reset_index(drop=True)
                            _render_momentum_table(category_frame, empty_message="No ticker momentum available for this category.")
                with category_history_tab:
                    detail_frame = result.monthly_ticker_frame.reset_index() if isinstance(result.monthly_ticker_frame.index, pd.MultiIndex) else result.monthly_ticker_frame.reset_index().rename(columns={"index": "ticker"})
                    left, right = st.columns(2)
                    with left:
                        category_history_months = int(st.selectbox("Lookback months", [6, 12, 18, 24], index=1, key="market_app::category_history_months"))
                    with right:
                        category_history_sort = st.selectbox("Sort mode", ["hierarchy", "last_month", "trailing_3m"], format_func=lambda value: {"hierarchy": "Category", "last_month": "Last month", "trailing_3m": "Trailing 3M"}[value], index=0, key="market_app::category_history_sort")
                    categories = detail_frame["category"].drop_duplicates().tolist() if "category" in detail_frame.columns else []
                    for category in categories:
                        with st.expander(str(category), expanded=False):
                            category_frame = detail_frame.loc[detail_frame["category"] == category, ["ticker", "description", *_monthly_history_columns(detail_frame)]].reset_index(drop=True)
                            category_frame, month_columns = _prepare_monthly_history_frame(category_frame, lookback_months=category_history_months, sort_mode="last_month" if category_history_sort == "hierarchy" else category_history_sort)
                            _render_monthly_history_table(category_frame, month_columns=month_columns, empty_message="No monthly history available for this category.")
            with tickers_tab:
                detail_momentum_tab, detail_history_tab = st.tabs(["Momentum", "Monthly history"])
                with detail_momentum_tab:
                    detail_frame = _ticker_detail_frame(result)
                    left, right = st.columns(2)
                    with left:
                        detail_ranking = st.selectbox("Ranking horizon", MOMENTUM_COLUMNS, format_func=lambda value: MOMENTUM_LABELS[value], index=0, key="market_app::detail_cat_ranking")
                    with right:
                        st.selectbox("Sort mode", ["last_month", "trailing_3m", "performance"], format_func=lambda value: {"last_month": "Last month", "trailing_3m": "Trailing 3M", "performance": "Performance"}[value], index=2, key="market_app::detail_cat_sort")
                    detail_frame = _sort_market_frame(detail_frame, detail_ranking, sort_mode="performance")
                    _render_momentum_table(detail_frame[["ticker", "description", *MOMENTUM_COLUMNS]], empty_message="No ticker detail available.")
                with detail_history_tab:
                    left, right = st.columns(2)
                    with left:
                        detail_history_months = int(st.selectbox("Lookback months", [6, 12, 18, 24], index=1, key="market_app::detail_cat_history_months"))
                    with right:
                        detail_history_sort = st.selectbox("Sort mode", ["last_month", "trailing_3m"], format_func=lambda value: {"last_month": "Last month", "trailing_3m": "Trailing 3M"}[value], index=0, key="market_app::detail_cat_history_sort")
                    detail_history_frame = _ticker_history_frame(result)
                    detail_history_frame = detail_history_frame.loc[:, ["ticker", "description", *_monthly_history_columns(detail_history_frame)]].reset_index(drop=True)
                    detail_history_frame, month_columns = _prepare_monthly_history_frame(detail_history_frame, lookback_months=detail_history_months, sort_mode=detail_history_sort)
                    _render_monthly_history_table(detail_history_frame, month_columns=month_columns)
            with ticker_tab:
                _render_ticker_view(result, key_prefix="market_app::category_ticker")
        else:
            overview_tab, tickers_tab, ticker_tab = st.tabs(["Overview", "Tickers", "Ticker"])
            with overview_tab:
                ranking_col, topn_col = st.columns(2)
                with ranking_col:
                    overview_ranking = st.selectbox(
                        "Ranking horizon",
                        MOMENTUM_COLUMNS,
                        format_func=lambda value: MOMENTUM_LABELS[value],
                        index=0,
                        key="market_app::overview_ranking",
                    )
                with topn_col:
                    overview_top_n = int(st.slider("Top / bottom rows", min_value=3, max_value=12, value=6, step=1, key="market_app::overview_top_n"))
                st.subheader("Market synthesis")
                st.caption(f"Quick read of the market ranked by {MOMENTUM_LABELS[overview_ranking].lower()} momentum.")
                _render_market_overview(result, ranking_column=overview_ranking, ranking_label=MOMENTUM_LABELS[overview_ranking], top_n=overview_top_n)
            with tickers_tab:
                detail_momentum_tab, detail_history_tab = st.tabs(["Momentum", "Monthly history"])
                with detail_momentum_tab:
                    detail_frame = _ticker_detail_frame(result)
                    left, right = st.columns(2)
                    with left:
                        detail_ranking = st.selectbox("Ranking horizon", MOMENTUM_COLUMNS, format_func=lambda value: MOMENTUM_LABELS[value], index=0, key="market_app::flat_detail_ranking")
                    with right:
                        st.selectbox("Sort mode", ["last_month", "trailing_3m", "performance"], format_func=lambda value: {"last_month": "Last month", "trailing_3m": "Trailing 3M", "performance": "Performance"}[value], index=2, key="market_app::flat_detail_sort")
                    detail_frame = _sort_market_frame(detail_frame, detail_ranking, sort_mode="performance")
                    _render_momentum_table(detail_frame[["ticker", "description", *MOMENTUM_COLUMNS]], empty_message="No ticker detail available.")
                with detail_history_tab:
                    left, right = st.columns(2)
                    with left:
                        detail_history_months = int(st.selectbox("Lookback months", [6, 12, 18, 24], index=1, key="market_app::flat_history_months"))
                    with right:
                        detail_history_sort = st.selectbox("Sort mode", ["last_month", "trailing_3m"], format_func=lambda value: {"last_month": "Last month", "trailing_3m": "Trailing 3M"}[value], index=0, key="market_app::flat_history_sort")
                    detail_history_frame = _ticker_history_frame(result)
                    detail_history_frame = detail_history_frame.loc[:, ["ticker", "description", *_monthly_history_columns(detail_history_frame)]].reset_index(drop=True)
                    detail_history_frame, month_columns = _prepare_monthly_history_frame(detail_history_frame, lookback_months=detail_history_months, sort_mode=detail_history_sort)
                    _render_monthly_history_table(detail_history_frame, month_columns=month_columns)
            with ticker_tab:
                _render_ticker_view(result, key_prefix="market_app::flat_ticker")
    elif section == "Config":
        _request_block(
            result.request,
            config_defaults,
            {
                "universe": result.universe,
                "start": result.start,
                "as_of_date": result.as_of_date,
                "synthesis_mode": result.synthesis_mode,
                "has_hierarchy": result.has_hierarchy,
                "num_tickers": int(len(result.ticker_frame)),
            },
        )
    else:
        _render_artifacts_block({name: str(path) for name, path in result.artifacts.files.items()})
