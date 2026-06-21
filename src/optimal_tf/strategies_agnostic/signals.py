from __future__ import annotations

from typing import Literal

import pandas as pd

from ..config import EstimationConfig
from ..strategies.common import resolve_allocation_date, sanitized_normalized_returns
from ..features import trend_ema_signal

SignalModel = Literal["ones", "trend_ema"]


def supported_signal_models() -> list[str]:
    """Return the dashboard/service-visible agnostic signal identifiers."""
    return ["ones", "trend_ema"]


def ones_signal_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp | str,
    assets: pd.Index | list[str] | None = None,
) -> pd.Series:
    """Return the constant ``p = 1`` signal on the available asset set."""
    del est_cfg
    ts = resolve_allocation_date(prices.index, as_of_date=date)
    asset_index = pd.Index(prices.columns if assets is None else assets)
    available = prices.loc[ts].reindex(asset_index).dropna().index
    signal = pd.Series(0.0, index=asset_index, dtype=float)
    signal.loc[available] = 1.0
    return signal


def trend_ema_signal_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp | str,
    assets: pd.Index | list[str] | None = None,
) -> pd.Series:
    """Return the current trend signal on volatility-normalized returns."""
    ts = resolve_allocation_date(prices.index, as_of_date=date)
    _, _, z_returns = sanitized_normalized_returns(prices.loc[prices.index <= ts], est_cfg)
    trend = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
    asset_index = pd.Index(prices.columns if assets is None else assets)
    if ts not in trend.index:
        return pd.Series(0.0, index=asset_index, dtype=float)
    return trend.loc[ts].reindex(asset_index).fillna(0.0).astype(float)


def resolve_signal(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp | str,
    corr: pd.DataFrame,
    signal_model: SignalModel,
) -> pd.Series:
    """Build one signal vector behind a single model-level interface."""
    if signal_model == "ones":
        return ones_signal_at_date(prices, est_cfg, date=date, assets=corr.index)
    if signal_model == "trend_ema":
        return trend_ema_signal_at_date(prices, est_cfg, date=date, assets=corr.index)
    raise ValueError(f"Unknown signal_model '{signal_model}'. Allowed values: {supported_signal_models()}.")


def resolve_signal_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp | str,
    corr: pd.DataFrame,
    signal_model: SignalModel,
) -> pd.DataFrame:
    """Return the history of `p_t` up to one allocation date on the current asset set."""
    ts = resolve_allocation_date(prices.index, as_of_date=date)
    asset_index = pd.Index(corr.index)
    history = prices.loc[prices.index <= ts]

    if signal_model == "ones":
        panel = pd.DataFrame(0.0, index=history.index, columns=asset_index, dtype=float)
        available = history.reindex(columns=asset_index).notna()
        panel = panel.where(~available, other=1.0)
        return panel

    if signal_model == "trend_ema":
        _, _, z_returns = sanitized_normalized_returns(history, est_cfg)
        trend = trend_ema_signal(z_returns, alpha=est_cfg.trend_alpha, span=est_cfg.trend_span)
        return trend.reindex(columns=asset_index).astype(float)

    raise ValueError(f"Unknown signal_model '{signal_model}'. Allowed values: {supported_signal_models()}.")
