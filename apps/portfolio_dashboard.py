from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from optimal_tf.market_fork import list_market_fork_snapshots
from optimal_tf.portfolio_explorer import (
    PortfolioExplorerContext,
    aggregate_current_weights,
    benchmark_tickers_for_level,
    current_holdings,
    equal_weight_returns,
    holdings_for_level,
    holdings_frame,
    load_portfolio_context,
    lookup_ticker_peers,
    selected_weight_history,
    sleeve_returns,
    trim_to_nav_start,
    trim_to_trading_start,
    window_return,
)
from trading_core.reporting import cumulative_nav

DEFAULT_FORK_DIR = "output/optimal_tf/market_forks"
LOOKBACK_OPTIONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "Since start": None,
}

st.set_page_config(page_title="portfolio dashboard", layout="wide")
st.title("portfolio dashboard")
st.caption("Explore a portfolio produced by an `optimal_tf` standard service with bucket-level EW references.")

MAX_CHART_POINTS = 750
EW_COLOR = "#111827"
EW_SECONDARY_COLOR = "#4b5563"
PORTFOLIO_COLOR = "#b45309"
TICKER_PALETTE = [
    "#1d4ed8",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#65a30d",
    "#9333ea",
    "#0f766e",
]


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


def _snapshot_choice_label(path: str) -> str:
    return Path(path).name


def _render_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No data available for this selection.")
        return
    table = frame.copy()
    for column in ("weight", "asset_return", "contribution_return", "active_return", "ew_return"):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce") * 100.0
    st.dataframe(table, width="stretch", hide_index=True)


def _drawdown_frame(nav_frame: pd.DataFrame) -> pd.DataFrame:
    if nav_frame.empty:
        return pd.DataFrame()
    return nav_frame.divide(nav_frame.cummax()).subtract(1.0)


def _nav_from_context_start(series: pd.Series, context: PortfolioExplorerContext) -> pd.Series:
    trimmed = trim_to_nav_start(series.fillna(0.0).astype(float), context)
    if trimmed.empty:
        return pd.Series(dtype=float)
    return cumulative_nav(trimmed)


def _display_chart_frame(frame: pd.DataFrame | pd.Series, *, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame | pd.Series:
    if frame.empty or len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    reduced = frame.iloc[::step].copy()
    if reduced.index[-1] != frame.index[-1]:
        reduced = pd.concat([reduced, frame.iloc[[-1]]])
        reduced = reduced[~reduced.index.duplicated(keep="last")]
    return reduced


def _is_ew_label(label: str) -> bool:
    normalized = str(label).strip().lower()
    return normalized.startswith("ew ")


def _ordered_columns(columns: list[str]) -> list[str]:
    ew_columns = [column for column in columns if _is_ew_label(column)]
    other_columns = [column for column in columns if column not in ew_columns]
    return [*ew_columns, *other_columns]


def _series_color_map(columns: list[str], *, ticker_order: list[str] | None = None) -> dict[str, str]:
    ordered = _ordered_columns(columns)
    mapping: dict[str, str] = {}
    palette_index = 0
    if "EW bucket" in ordered:
        mapping["EW bucket"] = EW_COLOR
    if "EW sector" in ordered:
        mapping["EW sector"] = EW_COLOR
    if "EW sub-sector" in ordered:
        mapping["EW sub-sector"] = EW_SECONDARY_COLOR
    if "Portfolio sleeve" in ordered:
        mapping["Portfolio sleeve"] = PORTFOLIO_COLOR

    ticker_names = ticker_order or []
    for ticker in ticker_names:
        if ticker in ordered and ticker not in mapping:
            mapping[ticker] = TICKER_PALETTE[palette_index % len(TICKER_PALETTE)]
            palette_index += 1
    for column in ordered:
        if column not in mapping:
            mapping[column] = TICKER_PALETTE[palette_index % len(TICKER_PALETTE)]
            palette_index += 1
    return mapping


def _render_line_chart(
    frame: pd.DataFrame | pd.Series,
    *,
    ticker_order: list[str] | None = None,
    height: int = 280,
) -> None:
    if isinstance(frame, pd.Series):
        plot = frame.to_frame()
    else:
        plot = frame.copy()
    if plot.empty:
        st.info("No data available for this chart.")
        return
    plot = _display_chart_frame(plot)
    plot = plot.loc[:, _ordered_columns(list(plot.columns))]
    colors = _series_color_map(list(plot.columns), ticker_order=ticker_order)
    long_frame = plot.reset_index(names="date").melt(id_vars="date", var_name="series", value_name="value")
    chart = (
        alt.Chart(long_frame)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title=None),
            color=alt.Color(
                "series:N",
                sort=_ordered_columns(list(plot.columns)),
                scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
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


@st.cache_data(show_spinner=False)
def _load_portfolio_context_cached(snapshot_path: str, refresh_policy: str = "auto") -> PortfolioExplorerContext:
    return load_portfolio_context(snapshot_path, refresh_policy=refresh_policy)


def _selection_nav_frame(
    context: PortfolioExplorerContext,
    *,
    level: str,
    selection: str,
    selected_tickers: list[str],
) -> pd.DataFrame:
    navs: dict[str, pd.Series] = {}
    if level == "ticker":
        ticker_returns = context.daily_asset_returns.loc[:, selected_tickers].iloc[:, 0].fillna(0.0)
        navs[selection] = _nav_from_context_start(ticker_returns, context)
        peers = lookup_ticker_peers(context, selection)
        if "sector" in peers and peers["sector"]:
            navs["EW sector"] = _nav_from_context_start(
                equal_weight_returns(context.daily_asset_returns, peers["sector"]),
                context,
            )
        if "sub_sector" in peers and peers["sub_sector"]:
            navs["EW sub-sector"] = _nav_from_context_start(
                equal_weight_returns(context.daily_asset_returns, peers["sub_sector"]),
                context,
            )
        return pd.DataFrame(navs)

    for ticker in selected_tickers:
        if ticker in context.daily_asset_returns.columns:
            navs[ticker] = _nav_from_context_start(context.daily_asset_returns[ticker], context)
    benchmark_tickers = benchmark_tickers_for_level(context, level, selection)
    if set(benchmark_tickers) == set(context.daily_asset_returns.columns) and not context.benchmark_returns.empty:
        navs["EW bucket"] = _nav_from_context_start(context.benchmark_returns, context)
    else:
        navs["EW bucket"] = _nav_from_context_start(
            equal_weight_returns(context.daily_asset_returns, benchmark_tickers),
            context,
        )
    return pd.DataFrame(navs)


def _selection_sleeve_nav_frame(
    context: PortfolioExplorerContext,
    *,
    level: str,
    selection: str,
    selected_tickers: list[str],
) -> pd.DataFrame:
    if level == "ticker":
        return _selection_nav_frame(context, level=level, selection=selection, selected_tickers=selected_tickers)

    full_universe = set(selected_tickers) == set(context.daily_asset_returns.columns)
    if full_universe:
        portfolio_returns = context.portfolio_returns_net
    else:
        portfolio_returns = sleeve_returns(context, selected_tickers)
    benchmark_tickers = benchmark_tickers_for_level(context, level, selection)
    if set(benchmark_tickers) == set(context.daily_asset_returns.columns) and not context.benchmark_returns.empty:
        benchmark_returns = context.benchmark_returns
    else:
        benchmark_returns = equal_weight_returns(context.daily_asset_returns, benchmark_tickers)
    return pd.DataFrame(
        {
            "Portfolio sleeve": _nav_from_context_start(portfolio_returns, context),
            "EW bucket": _nav_from_context_start(benchmark_returns, context),
        }
    )


def _portfolio_nav_frame(context: PortfolioExplorerContext) -> pd.DataFrame:
    portfolio_returns = context.portfolio_returns_net.fillna(0.0)
    if context.benchmark_returns.empty:
        ew_returns = equal_weight_returns(context.daily_asset_returns, list(context.daily_asset_returns.columns))
    else:
        ew_returns = context.benchmark_returns
    return pd.DataFrame(
        {
            "EW universe": _nav_from_context_start(ew_returns, context),
            "Portfolio": _nav_from_context_start(portfolio_returns, context),
        }
    )


def _portfolio_exposure_frame(context: PortfolioExplorerContext) -> pd.DataFrame:
    weights = trim_to_trading_start(context.daily_weights, context)
    if weights.empty:
        return pd.DataFrame()
    positive = weights.clip(lower=0.0).sum(axis=1).rename("Positive weights")
    negative = weights.clip(upper=0.0).sum(axis=1).rename("Negative weights")
    net = weights.sum(axis=1).rename("Net weights")
    return pd.concat([positive, negative, net], axis=1)


def _current_taxonomy(context: PortfolioExplorerContext) -> tuple[str, str | None, str, str | None]:
    frame = holdings_frame(context)
    has_sector = frame["sector"].astype(str).str.strip().ne("").any()
    has_sub_sector = frame["sub_sector"].astype(str).str.strip().ne("").any()
    if has_sector:
        return "sector", ("sub_sector" if has_sub_sector else None), "Sector", ("Sub-sector" if has_sub_sector else None)
    has_category = frame["category"].astype(str).str.strip().ne("").any()
    has_sub_category = frame["sub_category"].astype(str).str.strip().ne("").any()
    if has_category:
        return "category", ("sub_category" if has_sub_category else None), "Category", ("Sub-category" if has_sub_category else None)
    return "ticker", None, "Ticker", None


def _level_values(frame: pd.DataFrame, column: str) -> list[str]:
    values = frame[column].astype(str).str.strip()
    return sorted(values.loc[values.ne("")].unique().tolist())


def _bucket_detail_frame(
    context: PortfolioExplorerContext,
    *,
    selected_tickers: list[str],
    lookback_days: int | None,
    benchmark_tickers: list[str],
) -> pd.DataFrame:
    if not selected_tickers:
        return pd.DataFrame()
    frame = holdings_frame(context)
    frame = frame.loc[frame["ticker"].isin(selected_tickers)].copy()
    ew_return = window_return(equal_weight_returns(context.daily_asset_returns, benchmark_tickers), lookback_days=lookback_days)
    asset_returns: list[float] = []
    contribution_returns: list[float] = []
    active_returns: list[float] = []
    for ticker in frame["ticker"]:
        asset_return = window_return(context.daily_asset_returns[ticker], lookback_days=lookback_days)
        asset_returns.append(asset_return)
        contribution_returns.append(float(frame.loc[frame["ticker"] == ticker, "weight"].iloc[0]) * asset_return)
        active_returns.append(asset_return - ew_return)
    frame["asset_return"] = asset_returns
    frame["contribution_return"] = contribution_returns
    frame["ew_return"] = ew_return
    frame["active_return"] = active_returns
    return frame


def _render_bucket_view(
    context: PortfolioExplorerContext,
    *,
    title: str,
    level: str,
    selection: str,
    selected_tickers: list[str],
    lookback_days: int | None,
) -> None:
    benchmark_tickers = benchmark_tickers_for_level(context, level, selection)
    if level == "ticker":
        selection_weight = float(context.current_weights.get(selection, 0.0))
        sleeve_return_value = window_return(context.daily_asset_returns[selection], lookback_days=lookback_days)
    else:
        selection_weight = float(current_holdings(context).reindex(selected_tickers).fillna(0.0).sum())
        sleeve_return_value = window_return(sleeve_returns(context, selected_tickers), lookback_days=lookback_days)
    benchmark_return_value = window_return(equal_weight_returns(context.daily_asset_returns, benchmark_tickers), lookback_days=lookback_days)

    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selection", selection)
    c2.metric("Current weight", f"{selection_weight * 100.0:.2f}%")
    c3.metric("Selection return", f"{sleeve_return_value * 100.0:.2f}%")
    c4.metric("Active vs EW", f"{(sleeve_return_value - benchmark_return_value) * 100.0:.2f}%")
    st.caption(f"Selected tickers: {len(selected_tickers)} | EW benchmark return: {benchmark_return_value * 100.0:.2f}%")

    summary_tab, nav_tab, drawdown_tab = st.tabs(["Summary", "NAV", "Drawdown"])

    with summary_tab:
        detail_frame = _bucket_detail_frame(
            context,
            selected_tickers=selected_tickers,
            lookback_days=lookback_days,
            benchmark_tickers=benchmark_tickers,
        )
        _render_table(detail_frame)
        sleeve_nav_frame = _selection_sleeve_nav_frame(
            context,
            level=level,
            selection=selection,
            selected_tickers=selected_tickers,
        )
        if not sleeve_nav_frame.empty:
            st.caption("NAV du paquet")
            _render_line_chart(sleeve_nav_frame, ticker_order=selected_tickers)
        if level == "ticker":
            peers = lookup_ticker_peers(context, selection)
            peer_rows: list[dict[str, Any]] = []
            for peer_level, tickers in peers.items():
                peer_rows.append(
                    {
                        "reference": peer_level,
                        "num_tickers": len(tickers),
                        "ew_return": window_return(equal_weight_returns(context.daily_asset_returns, tickers), lookback_days=lookback_days),
                    }
                )
            if peer_rows:
                st.caption("Ticker references")
                _render_table(pd.DataFrame(peer_rows))

    with nav_tab:
        nav_frame = _selection_nav_frame(context, level=level, selection=selection, selected_tickers=selected_tickers)
        if nav_frame.empty:
            st.info("No NAV series available for this selection.")
        else:
            _render_line_chart(nav_frame, ticker_order=selected_tickers)
        st.caption("Positions over time")
        weights_history = selected_weight_history(context, selected_tickers)
        weights_history = trim_to_trading_start(weights_history, context)
        if weights_history.empty:
            st.info("No positions history available for this selection.")
        else:
            _render_line_chart(weights_history, ticker_order=selected_tickers)

    with drawdown_tab:
        drawdown_frame = _drawdown_frame(_selection_nav_frame(context, level=level, selection=selection, selected_tickers=selected_tickers))
        if drawdown_frame.empty:
            st.info("No drawdown series available for this selection.")
        else:
            _render_line_chart(drawdown_frame, ticker_order=selected_tickers)


snapshot_path_from_query = str(st.query_params.get("fork", ""))
default_snapshot_dir = str(Path(snapshot_path_from_query).parent) if snapshot_path_from_query else DEFAULT_FORK_DIR
snapshot_dir = st.sidebar.text_input("Fork snapshot dir", value=default_snapshot_dir)
recent_snapshot_paths = [str(path) for path in list_market_fork_snapshots(snapshot_dir)]
recent_snapshot_options = [""] + recent_snapshot_paths
default_recent_snapshot = snapshot_path_from_query if snapshot_path_from_query in recent_snapshot_options else (
    recent_snapshot_options[1] if len(recent_snapshot_options) > 1 else ""
)
selected_recent_snapshot = st.sidebar.selectbox(
    "Recent fork snapshots",
    recent_snapshot_options,
    index=recent_snapshot_options.index(default_recent_snapshot) if default_recent_snapshot in recent_snapshot_options else 0,
    format_func=lambda value: "None" if not value else _snapshot_choice_label(value),
)
snapshot_path = st.sidebar.text_input(
    "Fork snapshot path",
    value=selected_recent_snapshot or snapshot_path_from_query,
    help="Pick a recent snapshot above, or provide a custom JSON path.",
)

context: PortfolioExplorerContext | None = None
context_error: str | None = None
if snapshot_path:
    try:
        context = _load_portfolio_context_cached(snapshot_path)
    except Exception as exc:  # pragma: no cover
        context_error = str(exc)

if context_error:
    st.sidebar.error(context_error)

if context is None:
    st.info("Select a fork snapshot produced by `optimal_tf_dashboard` to explore a portfolio.")
    st.stop()

st.sidebar.caption(f"Loaded snapshot: {Path(context.snapshot_path).name}")
lookback_label = st.sidebar.selectbox("Return window", list(LOOKBACK_OPTIONS.keys()), index=1)
lookback_days = LOOKBACK_OPTIONS[lookback_label]

st.caption(
    f"Universe: `{context.universe}` | mode: `{context.mode}` | trading start: `{context.trading_start_date.date()}` | as of: `{context.as_of_date.date()}`"
)

primary_level, secondary_level, primary_label, secondary_label = _current_taxonomy(context)
holdings = holdings_frame(context)

portfolio_tab, primary_tab, secondary_tab, ticker_tab, context_tab = st.tabs(
    [
        "Portfolio",
        f"By {primary_label}",
        f"By {secondary_label or primary_label}",
        "By Ticker",
        "Context",
    ]
)

with portfolio_tab:
    portfolio_summary_tab, portfolio_nav_tab, portfolio_drawdown_tab = st.tabs(["Summary", "NAV", "Drawdown"])
    with portfolio_summary_tab:
        summary_frame = current_holdings(context).rename("weight").reset_index().rename(columns={"index": "ticker"})
        _render_table(summary_frame)
    with portfolio_nav_tab:
        portfolio_nav = _portfolio_nav_frame(context)
        _render_line_chart(portfolio_nav)
        st.caption("Exposition du portefeuille")
        exposure_frame = _portfolio_exposure_frame(context)
        if exposure_frame.empty:
            st.info("No exposure history available for this portfolio.")
        else:
            _render_line_chart(exposure_frame)
    with portfolio_drawdown_tab:
        portfolio_nav = _portfolio_nav_frame(context)
        _render_line_chart(_drawdown_frame(portfolio_nav))

with primary_tab:
    if primary_level == "ticker":
        st.info("No higher-level taxonomy available for this universe.")
    else:
        primary_values = _level_values(holdings, primary_level)
        selected_primary = st.selectbox(primary_label, primary_values, key="portfolio::primary")
        primary_tickers = holdings_for_level(context, primary_level, selected_primary)
        _render_bucket_view(
            context,
            title=f"{primary_label}: {selected_primary}",
            level=primary_level,
            selection=selected_primary,
            selected_tickers=primary_tickers,
            lookback_days=lookback_days,
        )

with secondary_tab:
    if primary_level == "ticker" or secondary_level is None:
        st.info("No secondary taxonomy available for this universe.")
    else:
        primary_values = _level_values(holdings, primary_level)
        selected_primary = st.selectbox(primary_label, primary_values, key="portfolio::secondary_primary")
        secondary_pool = holdings.loc[holdings[primary_level].astype(str) == selected_primary].copy()
        secondary_values = _level_values(secondary_pool, secondary_level)
        if not secondary_values:
            st.info("No secondary bucket available for this selection.")
        else:
            selected_secondary = st.selectbox(secondary_label or secondary_level, secondary_values, key="portfolio::secondary")
            secondary_tickers = secondary_pool.loc[
                secondary_pool[secondary_level].astype(str) == selected_secondary, "ticker"
            ].astype(str).tolist()
            _render_bucket_view(
                context,
                title=f"{secondary_label}: {selected_secondary}",
                level=secondary_level,
                selection=selected_secondary,
                selected_tickers=secondary_tickers,
                lookback_days=lookback_days,
            )

with ticker_tab:
    if primary_level == "ticker":
        ticker_values = _level_values(holdings, "ticker")
        selected_ticker = st.selectbox("Ticker", ticker_values, key="portfolio::ticker_only")
    elif secondary_level is None:
        primary_values = _level_values(holdings, primary_level)
        selected_primary = st.selectbox(primary_label, primary_values, key="portfolio::ticker_primary_only")
        ticker_pool = holdings.loc[holdings[primary_level].astype(str) == selected_primary].copy()
        ticker_values = _level_values(ticker_pool, "ticker")
        selected_ticker = st.selectbox("Ticker", ticker_values, key="portfolio::ticker_under_primary")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            primary_values = _level_values(holdings, primary_level)
            selected_primary = st.selectbox(primary_label, primary_values, key="portfolio::ticker_primary")
        ticker_pool = holdings.loc[holdings[primary_level].astype(str) == selected_primary].copy()
        with col2:
            secondary_values = _level_values(ticker_pool, secondary_level)
            selected_secondary = st.selectbox(secondary_label or secondary_level, secondary_values, key="portfolio::ticker_secondary")
        ticker_pool = ticker_pool.loc[ticker_pool[secondary_level].astype(str) == selected_secondary].copy()
        with col3:
            ticker_values = _level_values(ticker_pool, "ticker")
            selected_ticker = st.selectbox("Ticker", ticker_values, key="portfolio::ticker")
    _render_bucket_view(
        context,
        title=f"Ticker: {selected_ticker}",
        level="ticker",
        selection=selected_ticker,
        selected_tickers=[selected_ticker],
        lookback_days=lookback_days,
    )

with context_tab:
    meta = {
        "snapshot_path": context.snapshot_path,
        "mode": context.mode,
        "universe": context.universe,
        "start": context.start,
        "anchor_date": context.anchor_date,
        "trading_start_date": context.trading_start_date,
        "as_of_date": context.as_of_date,
        "source_service": context.snapshot.source_service,
        "label": context.snapshot.label,
    }
    st.caption("Portfolio context")
    st.json(_json_safe(meta), expanded=False)
    if context.snapshot.source_context:
        st.caption("Source context")
        st.json(_json_safe(context.snapshot.source_context), expanded=False)
    if context.snapshot.source_request:
        st.caption("Source request")
        st.json(_json_safe(context.snapshot.source_request), expanded=False)
