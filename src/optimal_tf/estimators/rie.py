from __future__ import annotations

import numpy as np
import pandas as pd

from .covariance import make_psd


def _renormalize_to_correlation(matrix: np.ndarray) -> np.ndarray:
    # Cleaning and eigenvalue flooring can slightly move the diagonal away from
    # one, so we project back to a proper correlation matrix here.
    diag = np.sqrt(np.clip(np.diag(matrix), 1e-12, None))
    corr = matrix / np.outer(diag, diag)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def _linear_shrinkage_correlation(corr: pd.DataFrame, shrinkage: float) -> pd.DataFrame:
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    arr = corr.to_numpy(dtype=float)
    # This gives us a stable baseline cleaner to validate the estimation
    # pipeline before the full nonlinear RIE shrinker is introduced.
    shrunk = (1.0 - shrinkage) * arr + shrinkage * np.eye(arr.shape[0])
    shrunk = _renormalize_to_correlation(shrunk)
    return pd.DataFrame(shrunk, index=corr.index, columns=corr.columns)


def clean_correlation_matrix(
    corr: pd.DataFrame,
    *,
    sample_size: int | None = None,
    method: str = "empirical",
    linear_shrinkage: float = 0.0,
    bandwidth: float = 1e-3,
) -> pd.DataFrame:
    """
    Clean a correlation matrix while preserving its eigenvectors.

    Notes
    -----
    The `rie` branch uses a rotation-invariant nonlinear shrinkage of the
    eigenvalues while keeping the sample eigenvectors fixed.
    """
    corr = make_psd(corr, floor=1e-10)
    corr = pd.DataFrame(_renormalize_to_correlation(corr.to_numpy(dtype=float)), index=corr.index, columns=corr.columns)

    if method == "empirical":
        return corr
    if method == "linear_shrinkage":
        return _linear_shrinkage_correlation(corr, linear_shrinkage)
    if method == "rie":
        vals, vecs = np.linalg.eigh(corr.to_numpy(dtype=float))
        vals = np.clip(vals, 1e-10, None)
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]

        p = len(vals)
        n = max(int(sample_size or p), 1)
        c = p / n

        # Use a small imaginary offset to approximate the Stieltjes transform
        # on the real axis while keeping the estimator numerically stable.
        eta = max(float(bandwidth) * max(float(np.median(vals)), 1.0), 1.0 / max(n, p))
        shrunk = np.empty_like(vals)
        for i, lam in enumerate(vals):
            z = lam - 1j * eta
            m = np.mean(1.0 / (vals - z))
            denom = abs(1.0 - c - c * lam * m) ** 2
            shrunk[i] = lam / max(denom, 1e-12)

        cleaned = vecs @ np.diag(np.clip(shrunk, 1e-10, None)) @ vecs.T
        cleaned = _renormalize_to_correlation(cleaned)
        return pd.DataFrame(cleaned, index=corr.index, columns=corr.columns)
    raise ValueError(f"Unknown cleaning method '{method}'")


def eigen_decomposition(corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(corr.to_numpy(dtype=float))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]
