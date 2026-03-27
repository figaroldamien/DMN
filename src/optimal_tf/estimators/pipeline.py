from __future__ import annotations

import pandas as pd

from ..config import EstimationConfig
from ..features import compute_returns, ewma_vol, normalize_returns_by_vol, rolling_corr_frame, sanitize_returns
from .covariance import correlation_to_covariance
from .rie import clean_correlation_matrix


def estimate_clean_covariance_panel(
    prices: pd.DataFrame,
    cfg: EstimationConfig,
) -> dict[pd.Timestamp, pd.DataFrame]:
    returns = sanitize_returns(compute_returns(prices), max_abs_return=cfg.max_abs_return)
    vol = ewma_vol(returns, span=cfg.vol_span)
    # The paper works with returns rescaled by realized volatility so that the
    # correlation cleaning focuses on cross-asset structure rather than scale.
    z_returns = normalize_returns_by_vol(returns, vol)
    raw_corr = rolling_corr_frame(
        z_returns,
        window=cfg.corr_span,
        min_periods=cfg.corr_min_periods,
    )

    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for ts, (corr, sample_size) in raw_corr.items():
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
