from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

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
from optimal_tf.data import load_prices_for_universe
from optimal_tf.market_fork import MarketForkSnapshot, load_market_fork_snapshot
from trading_core.backtest import slice_next_holding_period

PortfolioLevel = Literal["ticker", "sector", "sub_sector", "category", "sub_category"]

LEVEL_LABELS: dict[str, PortfolioLevel] = {
    "Ticker": "ticker",
    "Sector": "sector",
    "Sub-sector": "sub_sector",
    "Category": "category",
    "Sub-category": "sub_category",
}

UNIVERSE_COMPONENTS = {
    "nasdaq100": NASDAQ100_COMPONENTS,
    "cac40": CAC40_COMPONENTS,
    "dji": DJI_COMPONENTS,
    "sp500": SP500_COMPONENTS,
    "sbf120": SBF120_COMPONENTS,
    "eurostoxx50": EUROSTOXX50_COMPONENTS,
    "eurostoxx600": EUROSTOXX600_COMPONENTS,
    "index": INDEX_COMPONENTS,
    "futures": FUTURES_COMPONENTS,
    "dataset": DATASET_COMPONENTS,
    "dataset_all": DATASET_COMPONENTS,
    "table8_all": DATASET_COMPONENTS,
    "table_8": DATASET_COMPONENTS,
    "world_index": WORLD_INDEX_COMPONENTS,
}


@dataclass(frozen=True)
class PortfolioExplorerContext:
    snapshot_path: str
    snapshot: MarketForkSnapshot
    mode: str
    universe: str
    start: str | None
    as_of_date: pd.Timestamp
    anchor_date: pd.Timestamp
    trading_start_date: pd.Timestamp
    nav_start_date: pd.Timestamp
    prices: pd.DataFrame
    daily_asset_returns: pd.DataFrame
    metadata: pd.DataFrame
    current_weights: pd.Series
    weights_by_rebalance: pd.DataFrame
    daily_weights: pd.DataFrame
    ticker_portfolio_returns: pd.DataFrame
    portfolio_returns_net: pd.Series
    benchmark_returns: pd.Series


def _component_metadata(universe: str, tickers: list[str]) -> pd.DataFrame:
    components = UNIVERSE_COMPONENTS.get(universe, {})
    rows: list[dict[str, str]] = []
    for ticker in tickers:
        meta = components.get(str(ticker), {})
        rows.append(
            {
                "ticker": str(ticker),
                "sector": str(meta.get("sector", "") or "").strip(),
                "sub_sector": str(meta.get("sub_sector", "") or "").strip(),
                "category": str(meta.get("category", "") or "").strip(),
                "sub_category": str(meta.get("sub_category", "") or "").strip(),
                "description": str(meta.get("description", "") or "").strip(),
            }
        )
    frame = pd.DataFrame(rows).set_index("ticker")
    return frame.fillna("")


def _read_weights_by_rebalance(path: str | Path, columns: pd.Index) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    frame = frame.reindex(columns=columns).fillna(0.0).astype(float)
    return frame.sort_index()


def _read_allocation_weights(path: str | Path, columns: pd.Index) -> tuple[pd.Series, pd.Timestamp]:
    frame = pd.read_csv(path)
    if frame.empty or "ticker" not in frame.columns or "weight" not in frame.columns:
        raise ValueError(f"Unsupported allocation weights file: {path}")
    weight_series = frame.set_index("ticker")["weight"].astype(float).reindex(columns).fillna(0.0)
    date_value = frame["date"].iloc[0] if "date" in frame.columns else None
    if date_value is None or str(date_value).strip() == "":
        raise ValueError(f"Allocation weights file has no date column: {path}")
    return weight_series, pd.Timestamp(date_value)


def _read_return_series(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty:
        return pd.Series(dtype=float)
    series = frame.iloc[:, 0].astype(float).copy()
    series.index = pd.DatetimeIndex(series.index)
    return series.sort_index()


def _build_daily_weights_from_rebalance(
    prices_index: pd.DatetimeIndex,
    weights_by_rebalance: pd.DataFrame,
) -> pd.DataFrame:
    daily = pd.DataFrame(0.0, index=prices_index, columns=weights_by_rebalance.columns, dtype=float)
    for pos, rebalance_date in enumerate(weights_by_rebalance.index):
        current_weights = weights_by_rebalance.loc[rebalance_date].fillna(0.0).astype(float)
        next_rebalance = weights_by_rebalance.index[pos + 1] if pos + 1 < len(weights_by_rebalance.index) else None
        period_index = slice_next_holding_period(prices_index, rebalance_date, next_rebalance, None)
        if len(period_index) == 0:
            continue
        daily.loc[period_index] = current_weights.to_numpy(dtype=float)
    return daily


def _build_daily_weights_from_snapshot(
    prices_index: pd.DatetimeIndex,
    current_weights: pd.Series,
    anchor_date: pd.Timestamp,
) -> pd.DataFrame:
    daily = pd.DataFrame(0.0, index=prices_index, columns=current_weights.index, dtype=float)
    period_index = prices_index[prices_index > anchor_date]
    if len(period_index) > 0:
        daily.loc[period_index] = current_weights.astype(float).to_numpy(dtype=float)
    return daily


def load_portfolio_context(
    snapshot_path: str | Path,
    *,
    refresh_policy: str = "auto",
) -> PortfolioExplorerContext:
    snapshot = load_market_fork_snapshot(snapshot_path)
    prices = load_prices_for_universe(snapshot.market_universe, start=snapshot.market_start, refresh_policy=refresh_policy)
    as_of_date = pd.Timestamp(snapshot.market_as_of_date) if snapshot.market_as_of_date else pd.Timestamp(prices.index[-1])
    prices = prices.loc[prices.index <= as_of_date].ffill()
    if prices.empty:
        raise ValueError("No prices available for the selected portfolio context.")

    metadata = _component_metadata(snapshot.market_universe, list(prices.columns)).reindex(prices.columns).fillna("")
    daily_asset_returns = prices.pct_change().fillna(0.0)
    artifacts = snapshot.source_artifacts

    if "weights_by_rebalance" in artifacts:
        weights_by_rebalance = _read_weights_by_rebalance(artifacts["weights_by_rebalance"], prices.columns)
        if weights_by_rebalance.empty:
            raise ValueError("weights_by_rebalance artifact is empty.")
        current_weights = weights_by_rebalance.iloc[-1].astype(float)
        anchor_date = pd.Timestamp(weights_by_rebalance.index[-1])
        daily_weights = _build_daily_weights_from_rebalance(pd.DatetimeIndex(prices.index), weights_by_rebalance)
        mode = "dynamic"
    elif "weights_csv" in artifacts:
        current_weights, anchor_date = _read_allocation_weights(artifacts["weights_csv"], prices.columns)
        weights_by_rebalance = pd.DataFrame([current_weights], index=pd.DatetimeIndex([anchor_date]))
        daily_weights = _build_daily_weights_from_snapshot(pd.DatetimeIndex(prices.index), current_weights, anchor_date)
        mode = "snapshot"
    else:
        raise ValueError(
            "This fork snapshot does not expose portfolio weights artifacts. "
            "Supported services: Allocation, Evaluation, Strategy testbed."
        )

    ticker_portfolio_returns = daily_asset_returns.mul(daily_weights, axis=0)
    if "daily_returns_net" in artifacts:
        portfolio_returns_net = _read_return_series(artifacts["daily_returns_net"]).reindex(prices.index).fillna(0.0)
    else:
        portfolio_returns_net = ticker_portfolio_returns.sum(axis=1).fillna(0.0)
    if "benchmark_returns" in artifacts:
        benchmark_returns = _read_return_series(artifacts["benchmark_returns"]).reindex(prices.index).fillna(0.0)
    else:
        benchmark_returns = pd.Series(dtype=float)
    gross_exposure = daily_weights.abs().sum(axis=1)
    active_dates = gross_exposure.index[gross_exposure > 0.0]
    trading_start_date = pd.Timestamp(active_dates[0]) if len(active_dates) else pd.Timestamp(anchor_date)
    evaluation_start = snapshot.source_request.get("evaluation_start") if isinstance(snapshot.source_request, dict) else None
    if evaluation_start is not None and str(evaluation_start).strip() != "":
        nav_start_date = pd.Timestamp(evaluation_start)
    else:
        nav_start_date = pd.Timestamp(trading_start_date)
    return PortfolioExplorerContext(
        snapshot_path=str(snapshot_path),
        snapshot=snapshot,
        mode=mode,
        universe=snapshot.market_universe,
        start=snapshot.market_start,
        as_of_date=pd.Timestamp(as_of_date),
        anchor_date=pd.Timestamp(anchor_date),
        trading_start_date=trading_start_date,
        nav_start_date=nav_start_date,
        prices=prices,
        daily_asset_returns=daily_asset_returns,
        metadata=metadata,
        current_weights=current_weights.reindex(prices.columns).fillna(0.0).astype(float),
        weights_by_rebalance=weights_by_rebalance.reindex(columns=prices.columns).fillna(0.0).astype(float),
        daily_weights=daily_weights.reindex(columns=prices.columns).fillna(0.0).astype(float),
        ticker_portfolio_returns=ticker_portfolio_returns.reindex(columns=prices.columns).fillna(0.0).astype(float),
        portfolio_returns_net=portfolio_returns_net.astype(float),
        benchmark_returns=benchmark_returns.astype(float),
    )


def current_holdings(context: PortfolioExplorerContext, *, min_abs_weight: float = 1e-8) -> pd.Series:
    weights = context.current_weights.astype(float).fillna(0.0)
    return weights.loc[weights.abs() > min_abs_weight].sort_values(key=lambda series: series.abs(), ascending=False)


def holdings_frame(context: PortfolioExplorerContext) -> pd.DataFrame:
    weights = current_holdings(context)
    frame = context.metadata.reindex(weights.index).copy()
    frame.insert(0, "ticker", weights.index.astype(str))
    frame.insert(1, "weight", weights.to_numpy(dtype=float))
    return frame.reset_index(drop=True)


def available_level_values(context: PortfolioExplorerContext, level: PortfolioLevel) -> list[str]:
    frame = holdings_frame(context)
    if level == "ticker":
        return frame["ticker"].astype(str).tolist()
    values = frame[level].astype(str).str.strip()
    values = values.loc[values.ne("")]
    return sorted(values.unique().tolist())


def holdings_for_level(context: PortfolioExplorerContext, level: PortfolioLevel, selection: str) -> list[str]:
    frame = holdings_frame(context)
    if level == "ticker":
        return [selection] if selection in set(frame["ticker"]) else []
    selected = frame.loc[frame[level].astype(str) == str(selection), "ticker"].astype(str)
    return selected.tolist()


def benchmark_tickers_for_level(context: PortfolioExplorerContext, level: PortfolioLevel, selection: str) -> list[str]:
    metadata = context.metadata.copy()
    if level == "ticker":
        if selection not in metadata.index:
            return []
        return [selection]
    selected = metadata.loc[metadata[level].astype(str) == str(selection)]
    return selected.index.astype(str).tolist()


def aggregate_current_weights(context: PortfolioExplorerContext, level: PortfolioLevel) -> pd.Series:
    frame = holdings_frame(context)
    if level == "ticker":
        return frame.set_index("ticker")["weight"].sort_values(key=lambda series: series.abs(), ascending=False)
    grouped = frame.groupby(level, dropna=False)["weight"].sum()
    grouped.index = grouped.index.astype(str)
    non_empty_mask = pd.Index(grouped.index).astype(str).str.strip() != ""
    grouped = grouped.loc[non_empty_mask]
    return grouped.sort_values(key=lambda series: series.abs(), ascending=False)


def rebased_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    filled = prices.ffill().copy()
    first_valid = filled.apply(lambda column: column.dropna().iloc[0] if not column.dropna().empty else np.nan)
    return 100.0 * filled.divide(first_valid.replace(0.0, np.nan), axis="columns")


def equal_weight_returns(daily_asset_returns: pd.DataFrame, tickers: list[str]) -> pd.Series:
    cols = [ticker for ticker in tickers if ticker in daily_asset_returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    return daily_asset_returns.loc[:, cols].mean(axis=1).fillna(0.0).astype(float)


def sleeve_returns(context: PortfolioExplorerContext, tickers: list[str]) -> pd.Series:
    cols = [ticker for ticker in tickers if ticker in context.ticker_portfolio_returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    numerator = context.ticker_portfolio_returns.loc[:, cols].sum(axis=1)
    gross = context.daily_weights.loc[:, cols].abs().sum(axis=1)
    normalized = numerator.divide(gross.replace(0.0, np.nan)).fillna(0.0)
    return normalized.astype(float)


def window_return(series: pd.Series, *, lookback_days: int | None = None) -> float:
    clean = series.dropna().astype(float)
    if clean.empty:
        return 0.0
    if lookback_days is not None and lookback_days > 0 and len(clean) > lookback_days:
        clean = clean.iloc[-lookback_days:]
    return float((1.0 + clean).prod() - 1.0)


def lookup_ticker_peers(context: PortfolioExplorerContext, ticker: str) -> dict[str, list[str]]:
    if ticker not in context.metadata.index:
        return {}
    row = context.metadata.loc[ticker]
    peers: dict[str, list[str]] = {}
    for level in ("sector", "sub_sector", "category", "sub_category"):
        value = str(row.get(level, "") or "").strip()
        if not value:
            continue
        peers[level] = benchmark_tickers_for_level(context, level, value)
    return peers


def trim_to_trading_start(frame: pd.DataFrame | pd.Series, context: PortfolioExplorerContext) -> pd.DataFrame | pd.Series:
    if frame.empty:
        return frame
    return frame.loc[frame.index >= context.trading_start_date]


def trim_to_nav_start(frame: pd.DataFrame | pd.Series, context: PortfolioExplorerContext) -> pd.DataFrame | pd.Series:
    if frame.empty:
        return frame
    return frame.loc[frame.index >= context.nav_start_date]


def selected_weight_history(context: PortfolioExplorerContext, tickers: list[str], *, normalize: bool = False) -> pd.DataFrame:
    cols = [ticker for ticker in tickers if ticker in context.weights_by_rebalance.columns]
    if not cols:
        return pd.DataFrame(index=context.weights_by_rebalance.index)
    if context.mode == "dynamic":
        history = context.weights_by_rebalance.loc[:, cols].copy()
    else:
        history = context.daily_weights.loc[:, cols].copy()
    if normalize:
        gross = history.abs().sum(axis=1)
        history = history.divide(gross.replace(0.0, np.nan), axis=0).fillna(0.0)
    return history.astype(float)
