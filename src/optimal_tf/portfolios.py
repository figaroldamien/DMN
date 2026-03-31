from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators.covariance import covariance_to_correlation
from .estimators.rie import eigen_decomposition


def normalize_weights(weights: pd.Series, *, long_only: bool = False) -> pd.Series:
    weights = weights.astype(float).fillna(0.0)
    if long_only:
        weights = weights.clip(lower=0.0)
        total = float(weights.sum())
        return weights / total if total > 0 else weights
    gross = float(weights.abs().sum())
    return weights / gross if gross > 0 else weights


def risk_parity_weights_from_cov(cov: pd.DataFrame) -> pd.Series:
    # This first version uses inverse-vol weights as the simplest cross-asset RP
    # baseline. We can later replace it with an equal-risk-contribution solver.
    vol = pd.Series(np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), 1e-12, None)), index=cov.index)
    weights = 1.0 / vol
    return weights / weights.abs().sum()


def risk_parity_weights_from_cov_with_tilt(
    cov: pd.DataFrame,
    tilt: pd.Series,
) -> pd.Series:
    vol = pd.Series(np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), 1e-12, None)), index=cov.index)
    aligned_tilt = tilt.reindex(cov.index).fillna(0.0).astype(float)
    raw = aligned_tilt / vol
    return normalize_weights(raw, long_only=False)


def agnostic_risk_parity_weights_from_cov(cov: pd.DataFrame) -> pd.Series:
    corr = covariance_to_correlation(cov)
    vals, vecs = eigen_decomposition(corr)
    # Whitening the correlation spectrum gives each decorrelated mode the same
    # ex-ante risk budget, which is the core intuition behind ARP.
    inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(vals, 1e-8, None)))
    whitened = vecs @ inv_sqrt @ vecs.T
    ones = np.ones(len(corr))
    weights = whitened @ ones
    series = pd.Series(weights, index=corr.index, dtype=float)
    return series / series.abs().sum()


def naive_markowitz_weights_from_cov(cov: pd.DataFrame) -> pd.Series:
    cov_arr = cov.to_numpy(dtype=float)
    # "Naive Markowitz" uses a flat expected return vector and leaves the
    # covariance matrix to determine the relative allocations.
    raw = np.linalg.pinv(cov_arr) @ np.ones(cov_arr.shape[0], dtype=float)
    return normalize_weights(pd.Series(raw, index=cov.index, dtype=float), long_only=False)


def trend_on_risk_parity_weights_from_cov_and_signal(
    cov: pd.DataFrame,
    signal: pd.Series,
    *,
    long_only: bool = False,
) -> pd.Series:
    # In the paper-faithful ToRP intuition, the trend is measured on the RP
    # factor itself and then applied back onto the RP portfolio.
    rp_weights = risk_parity_weights_from_cov(cov)
    aligned_signal = signal.reindex(rp_weights.index).fillna(0.0).astype(float)
    rp_factor_signal = float(aligned_signal @ rp_weights)
    raw = rp_weights * rp_factor_signal
    return normalize_weights(raw, long_only=long_only)


def trend_on_risk_parity_weights_from_cov_and_factor_signal(
    cov: pd.DataFrame,
    factor_signal: float,
    *,
    tilt: pd.Series | None = None,
    long_only: bool = False,
) -> pd.Series:
    if tilt is None:
        rp_weights = risk_parity_weights_from_cov(cov)
    else:
        rp_weights = risk_parity_weights_from_cov_with_tilt(cov, tilt)
    raw = rp_weights * float(factor_signal)
    return normalize_weights(raw, long_only=long_only)
