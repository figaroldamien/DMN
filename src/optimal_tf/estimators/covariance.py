from __future__ import annotations

import numpy as np
import pandas as pd


def covariance_to_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    # Correlations are built from the covariance diagonal, with a small floor to
    # avoid exploding ratios when an asset has vanishing estimated variance.
    std = np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), 1e-12, None))
    denom = np.outer(std, std)
    corr = cov.to_numpy(dtype=float) / denom
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def correlation_to_covariance(corr: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    # We rebuild the covariance after correlation cleaning so vol estimation and
    # cross-sectional cleaning remain two separate, swappable steps.
    sigma = vol.reindex(corr.index).to_numpy(dtype=float)
    cov = corr.to_numpy(dtype=float) * np.outer(sigma, sigma)
    return pd.DataFrame(cov, index=corr.index, columns=corr.columns)


def make_psd(matrix: pd.DataFrame, floor: float = 1e-8) -> pd.DataFrame:
    # Flooring eigenvalues is the simplest way to restore a usable PSD matrix
    # before a more sophisticated cleaner such as RIE is applied.
    vals, vecs = np.linalg.eigh(matrix.to_numpy(dtype=float))
    vals = np.maximum(vals, floor)
    fixed = vecs @ np.diag(vals) @ vecs.T
    fixed = 0.5 * (fixed + fixed.T)
    return pd.DataFrame(fixed, index=matrix.index, columns=matrix.columns)
