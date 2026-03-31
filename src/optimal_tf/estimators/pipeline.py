from __future__ import annotations

import pandas as pd

from ..config import EstimationConfig
from ..features import alpha_from_span, compute_returns, ewma_cov_frame, ewma_vol, normalize_returns_by_vol, sanitize_returns
from .covariance import correlation_to_covariance, covariance_to_correlation
from .rie import clean_correlation_matrix


def estimate_clean_covariance_at_date(
    prices: pd.DataFrame,
    cfg: EstimationConfig,
    date: pd.Timestamp | str,
) -> pd.DataFrame:
    ts = pd.Timestamp(date)
    history = prices.loc[prices.index <= ts]
    if history.empty:
        raise ValueError(f"No price history available on or before {ts.date()}.")
    panel = estimate_clean_covariance_panel(history, cfg)
    if not panel:
        raise ValueError(f"Not enough history to estimate covariance on {ts.date()}.")
    eligible = [key for key in panel if key <= ts]
    if not eligible:
        raise ValueError(f"No covariance estimate available on or before {ts.date()}.")
    return panel[max(eligible)]


def estimate_clean_covariance_panel(
    prices: pd.DataFrame,
    cfg: EstimationConfig,
) -> dict[pd.Timestamp, pd.DataFrame]:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=cfg.max_abs_return)
    vol = ewma_vol(returns, span=cfg.vol_span)
    # The paper works with returns rescaled by realized volatility so that the
    # correlation cleaning focuses on cross-asset structure rather than scale.
    z_returns = normalize_returns_by_vol(returns, vol)
    covariance_alpha = cfg.covariance_alpha
    if covariance_alpha is None:
        covariance_alpha = alpha_from_span(cfg.corr_span)
    raw_cov = ewma_cov_frame(
        z_returns,
        alpha=float(covariance_alpha),
        min_periods=cfg.covariance_min_periods,
    )

    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for ts, (cov_z, sample_size) in raw_cov.items():
        corr = covariance_to_correlation(cov_z)
        clean_corr = clean_correlation_matrix(
            corr,
            sample_size=sample_size,
            method=cfg.cleaning_method,
            linear_shrinkage=cfg.linear_shrinkage,
            bandwidth=cfg.rie_bandwidth,
        )
        vol_t = vol.loc[ts].dropna()
        tickers = [ticker for ticker in clean_corr.index if ticker in vol_t.index]
        if not tickers:
            continue
        # We rebuild the covariance on the intersection so downstream portfolio
        # builders always receive aligned square matrices.
        out[ts] = correlation_to_covariance(clean_corr.loc[tickers, tickers], vol_t.loc[tickers])
    return out
