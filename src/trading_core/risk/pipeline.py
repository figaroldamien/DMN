from __future__ import annotations

import pandas as pd

from trading_core.features import compute_returns, effective_span_from_alpha, ewma_vol, normalize_returns_by_vol, sanitize_returns

from .covariance import correlation_to_covariance
from .rie import clean_correlation_matrix


def _resolve_covariance_window(cfg) -> int:
    if cfg.covariance_window is not None:
        if cfg.covariance_window <= 0:
            raise ValueError("covariance_window must be strictly positive.")
        return int(cfg.covariance_window)
    if cfg.corr_span is not None and cfg.corr_span > 0:
        return int(cfg.corr_span)
    if cfg.covariance_alpha is not None:
        resolved = effective_span_from_alpha(cfg.covariance_alpha)
        if resolved is None:
            raise ValueError("covariance_alpha could not be converted to a window.")
        return resolved
    raise ValueError("One of covariance_window, corr_span, or covariance_alpha must be provided.")


def _build_risk_inputs(
    prices: pd.DataFrame,
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=cfg.max_abs_return)
    vol = ewma_vol(returns, span=cfg.vol_span)
    z_returns = normalize_returns_by_vol(returns, vol)
    return returns, vol, z_returns


def rolling_corr_frame(
    frame: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    target_dates: pd.Index | None = None,
) -> dict[pd.Timestamp, tuple[pd.DataFrame, int, pd.DataFrame]]:
    min_periods = window if min_periods is None else min_periods
    out: dict[pd.Timestamp, tuple[pd.DataFrame, int, pd.DataFrame]] = {}
    if target_dates is None:
        positions = range(len(frame))
    else:
        target_index = pd.DatetimeIndex(target_dates)
        resolved_positions = frame.index.get_indexer(target_index)
        positions = sorted({int(idx) for idx in resolved_positions if idx >= 0})
    for idx in positions:
        end = frame.index[idx]
        sample = frame.iloc[max(0, idx - window + 1) : idx + 1]
        valid_cols = sample.columns[sample.notna().sum(axis=0) >= min_periods]
        if len(valid_cols) < 2:
            continue
        sample = sample.loc[:, valid_cols]
        corr = sample.corr(min_periods=min_periods)
        keep = corr.index[~corr.isna().any(axis=1)]
        corr = corr.loc[keep, keep]
        if corr.empty or len(corr) < 2:
            continue
        sample_size = int(sample.loc[:, keep].notna().sum(axis=0).min())
        out[end] = (corr.astype(float), sample_size, sample.loc[:, keep].astype(float))
    return out


def estimate_clean_covariance_at_date(
    prices: pd.DataFrame,
    cfg,
    date: pd.Timestamp | str,
) -> pd.DataFrame:
    ts = pd.Timestamp(date)
    history = prices.loc[prices.index <= ts]
    if history.empty:
        raise ValueError(f"No price history available on or before {ts.date()}.")
    panel = estimate_clean_covariance_panel(history, cfg, target_dates=pd.DatetimeIndex([ts]))
    if not panel:
        raise ValueError(f"Not enough history to estimate covariance on {ts.date()}.")
    eligible = [key for key in panel if key <= ts]
    if not eligible:
        raise ValueError(f"No covariance estimate available on or before {ts.date()}.")
    return panel[max(eligible)]


def estimate_clean_correlation_at_date(
    prices: pd.DataFrame,
    cfg,
    date: pd.Timestamp | str,
) -> pd.DataFrame:
    ts = pd.Timestamp(date)
    history = prices.loc[prices.index <= ts]
    if history.empty:
        raise ValueError(f"No price history available on or before {ts.date()}.")
    panel = estimate_clean_correlation_panel(history, cfg, target_dates=pd.DatetimeIndex([ts]))
    if not panel:
        raise ValueError(f"Not enough history to estimate correlation on {ts.date()}.")
    eligible = [key for key in panel if key <= ts]
    if not eligible:
        raise ValueError(f"No correlation estimate available on or before {ts.date()}.")
    return panel[max(eligible)]


def estimate_clean_correlation_panel(
    prices: pd.DataFrame,
    cfg,
    target_dates: pd.Index | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    _, _, z_returns = _build_risk_inputs(prices, cfg)
    return estimate_clean_correlation_panel_from_z_returns(z_returns, cfg, target_dates=target_dates)


def estimate_clean_correlation_panel_from_z_returns(
    z_returns: pd.DataFrame,
    cfg,
    target_dates: pd.Index | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    covariance_window = _resolve_covariance_window(cfg)
    raw_corr = rolling_corr_frame(
        z_returns,
        window=covariance_window,
        min_periods=cfg.covariance_min_periods,
        target_dates=target_dates,
    )

    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for ts, (corr, sample_size, sample_frame) in raw_corr.items():
        out[ts] = clean_correlation_matrix(
            corr,
            data=sample_frame,
            sample_size=sample_size,
            method=cfg.cleaning_method,
            linear_shrinkage=cfg.linear_shrinkage,
            bandwidth=cfg.rie_bandwidth,
        )
    return out


def estimate_clean_covariance_panel(
    prices: pd.DataFrame,
    cfg,
    target_dates: pd.Index | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    _, vol, z_returns = _build_risk_inputs(prices, cfg)
    corr_panel = estimate_clean_correlation_panel_from_z_returns(
        z_returns,
        cfg,
        target_dates=target_dates,
    )

    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for ts, clean_corr in corr_panel.items():
        vol_t = vol.loc[ts].dropna()
        tickers = [ticker for ticker in clean_corr.index if ticker in vol_t.index]
        if not tickers:
            continue
        out[ts] = correlation_to_covariance(clean_corr.loc[tickers, tickers], vol_t.loc[tickers])
    return out
