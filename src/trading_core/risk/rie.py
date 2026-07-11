from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from .covariance import make_psd

SUPPORTED_CLEANING_METHODS = ("empirical", "linear_shrinkage", "rie_spectral", "rie_reference", "rie")
_EIGENVALUE_FLOOR = 1e-10
_DENOMINATOR_FLOOR = 1e-12


Cleaner = Callable[..., pd.DataFrame]


def _renormalize_to_correlation(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix back to a proper correlation matrix."""
    diag = np.sqrt(np.clip(np.diag(matrix), _DENOMINATOR_FLOOR, None))
    corr = matrix / np.outer(diag, diag)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def _to_frame(matrix: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Wrap a dense matrix with the same row/column labels."""
    return pd.DataFrame(matrix, index=index, columns=index)


def _normalize_input_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    """Stabilize the input before any cleaning-specific transformation."""
    corr = make_psd(corr, floor=_EIGENVALUE_FLOOR)
    normalized = _renormalize_to_correlation(corr.to_numpy(dtype=float))
    return _to_frame(normalized, corr.index)


def _linear_shrinkage_correlation(corr: pd.DataFrame, shrinkage: float) -> pd.DataFrame:
    """Blend the empirical correlation with the identity matrix."""
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    arr = corr.to_numpy(dtype=float)
    shrunk = (1.0 - shrinkage) * arr + shrinkage * np.eye(arr.shape[0])
    return _to_frame(_renormalize_to_correlation(shrunk), corr.index)


def supported_cleaning_methods() -> tuple[str, ...]:
    return SUPPORTED_CLEANING_METHODS


def _reference_rie_correlation(data: pd.DataFrame, *, normalize: bool) -> pd.DataFrame:
    """Run the optional external benchmark implementation on the source data."""
    try:
        rie_estimator = importlib.import_module("rie_estimator")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'rie_estimator' package is required for cleaning_method='rie_reference'. "
            "Install it with: pip install rie-estimator"
        ) from exc

    clean_input = data.astype(float)
    clean_input = clean_input.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if clean_input.empty or clean_input.shape[1] < 2:
        raise ValueError("rie_reference requires at least two assets and one complete observation row.")

    cleaned = rie_estimator.get_rie(clean_input, normalize=normalize, max_ones=True)
    cleaned = np.asarray(cleaned, dtype=float)
    cleaned = _renormalize_to_correlation(cleaned)
    return _to_frame(cleaned, clean_input.columns)


def _estimate_stieltjes_transform(eigenvalues: np.ndarray, z: complex) -> complex:
    """Estimate the empirical Stieltjes transform at one complex evaluation point."""
    denom_terms = eigenvalues - z
    inv_terms = np.divide(
        1.0,
        denom_terms,
        out=np.zeros_like(denom_terms, dtype=np.complex128),
        where=np.abs(denom_terms) > 1e-14,
    )
    finite_terms = inv_terms[np.isfinite(inv_terms)]
    if len(finite_terms) == 0:
        return 0.0 + 0.0j
    return np.mean(finite_terms)


def _native_rie_shrunk_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    sample_size: int | None,
    bandwidth: float,
) -> np.ndarray:
    """Apply the native nonlinear shrinkage formula to empirical eigenvalues.

    This is a lightweight in-house implementation used for experiments and
    internal comparisons. It keeps the empirical eigenvectors and only adjusts
    the eigenvalue spectrum.
    """
    p = len(eigenvalues)
    n = max(int(sample_size or p), 1)
    concentration = p / n

    # The complex shift controls the local smoothing used by the Stieltjes
    # transform estimate. We keep a small floor so the estimate remains usable
    # on small samples and nearly degenerate spectra.
    eta = max(float(bandwidth) * max(float(np.median(eigenvalues)), 1.0), 1.0 / max(n, p))
    shrunk = np.empty_like(eigenvalues)
    for i, eigenvalue in enumerate(eigenvalues):
        z = eigenvalue - 1j * eta
        stieltjes = _estimate_stieltjes_transform(eigenvalues, z)
        denominator = abs(1.0 - concentration - concentration * eigenvalue * stieltjes) ** 2
        shrunk[i] = eigenvalue / max(denominator, _DENOMINATOR_FLOOR)
    return np.clip(shrunk, _EIGENVALUE_FLOOR, None)


def _native_rie_correlation(
    corr: pd.DataFrame,
    *,
    sample_size: int | None,
    bandwidth: float,
) -> pd.DataFrame:
    """Clean a correlation matrix with the native RIE approximation."""
    eigenvalues, eigenvectors = np.linalg.eigh(corr.to_numpy(dtype=float))
    eigenvalues = np.clip(eigenvalues, _EIGENVALUE_FLOOR, None)
    order = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[order]
    sorted_eigenvectors = eigenvectors[:, order]

    shrunk = _native_rie_shrunk_eigenvalues(
        sorted_eigenvalues,
        sample_size=sample_size,
        bandwidth=bandwidth,
    )
    cleaned = sorted_eigenvectors @ np.diag(shrunk) @ sorted_eigenvectors.T
    cleaned = _renormalize_to_correlation(cleaned)
    return _to_frame(cleaned, corr.index)


def _spectral_rie_correlation(
    corr: pd.DataFrame,
    *,
    sample_size: int | None,
) -> pd.DataFrame:
    """Clean a correlation matrix with the in-project spectral RIE implementation."""
    if sample_size is None:
        raise ValueError("clean_correlation_matrix(method='rie_spectral') requires sample_size.")

    from optimal_tf.estimators.rie_spectral import clean_correlation_matrix_rie_spectral

    result = clean_correlation_matrix_rie_spectral(corr, sample_size=int(sample_size))
    return result.cleaned_matrix


def clean_correlation_matrix(
    corr: pd.DataFrame,
    *,
    data: pd.DataFrame | None = None,
    sample_size: int | None = None,
    method: str = "empirical",
    linear_shrinkage: float = 0.0,
    bandwidth: float = 1e-3,
) -> pd.DataFrame:
    """Clean a correlation matrix with one of the supported estimators."""
    normalized_corr = _normalize_input_correlation(corr)

    if method == "empirical":
        return normalized_corr
    if method == "linear_shrinkage":
        return _linear_shrinkage_correlation(normalized_corr, linear_shrinkage)
    if method == "rie_spectral":
        return _spectral_rie_correlation(normalized_corr, sample_size=sample_size)
    if method == "rie":
        return _spectral_rie_correlation(normalized_corr, sample_size=sample_size)
    if method == "rie_reference":
        if data is None:
            raise ValueError("clean_correlation_matrix(method='rie_reference') requires the source data frame.")
        return _reference_rie_correlation(data, normalize=False).reindex(index=normalized_corr.index, columns=normalized_corr.columns)
    raise ValueError(f"Unknown cleaning method '{method}'")


def eigen_decomposition(corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors ordered from largest to smallest."""
    vals, vecs = np.linalg.eigh(corr.to_numpy(dtype=float))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]
