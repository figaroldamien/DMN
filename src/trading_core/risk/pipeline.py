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


def rolling_corr_frame(
    frame: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    target_dates: pd.Index | None = None,
) -> dict[pd.Timestamp, tuple[pd.DataFrame, int, pd.DataFrame]]:
    min_periods = window if min_periods is None else min_periods
    target_set = None if target_dates is None else set(pd.DatetimeIndex(target_dates))
    out: dict[pd.Timestamp, tuple[pd.DataFrame, int, pd.DataFrame]] = {}
    for idx in range(len(frame)):
        end = frame.index[idx]
        if target_set is not None and end not in target_set:
            continue
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


def estimate_clean_covariance_panel(
    prices: pd.DataFrame,
    cfg,
    target_dates: pd.Index | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=cfg.max_abs_return)
    vol = ewma_vol(returns, span=cfg.vol_span)
    z_returns = normalize_returns_by_vol(returns, vol)
    covariance_window = _resolve_covariance_window(cfg)
    raw_corr = rolling_corr_frame(
        z_returns,
        window=covariance_window,
        min_periods=cfg.covariance_min_periods,
        target_dates=target_dates,
    )

    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for ts, (corr, sample_size, sample_frame) in raw_corr.items():
        clean_corr = clean_correlation_matrix(
            corr,
            data=sample_frame,
            sample_size=sample_size,
            method=cfg.cleaning_method,
            linear_shrinkage=cfg.linear_shrinkage,
            bandwidth=cfg.rie_bandwidth,
        )
        vol_t = vol.loc[ts].dropna()
        tickers = [ticker for ticker in clean_corr.index if ticker in vol_t.index]
        if not tickers:
            continue
        out[ts] = correlation_to_covariance(clean_corr.loc[tickers, tickers], vol_t.loc[tickers])
    return out
