from __future__ import annotations

import pandas as pd

from trading_core.features import (
    alpha_from_span,
    compute_returns,
    effective_span_from_alpha,
    ewma_cov_frame,
    ewma_vol,
    normalize_returns_by_vol,
    resolve_ewma_alpha,
    sanitize_returns,
    trend_ema_signal,
)


def rolling_corr_frame(
    frame: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    target_dates: pd.Index | None = None,
) -> dict[pd.Timestamp, tuple[pd.DataFrame, int]]:
    # This helper remains local to optimal_tf for now because the staggered-
    # universe rolling correlation behavior was introduced here first during the
    # SP500 covariance fix. The rest of the feature primitives live in
    # trading_core.features.
    min_periods = window if min_periods is None else min_periods
    target_set = None if target_dates is None else set(pd.DatetimeIndex(target_dates))
    out: dict[pd.Timestamp, tuple[pd.DataFrame, int]] = {}
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
        out[end] = (corr.astype(float), sample_size)
    return out


__all__ = [
    "alpha_from_span",
    "compute_returns",
    "effective_span_from_alpha",
    "ewma_cov_frame",
    "ewma_vol",
    "normalize_returns_by_vol",
    "resolve_ewma_alpha",
    "rolling_corr_frame",
    "sanitize_returns",
    "trend_ema_signal",
]
