from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_core.risk.covariance import make_psd
from trading_core.risk.rie import clean_correlation_matrix

from .rie_spectral import (
    RieCorrelationResult,
    RieSpectralDecomposition,
    clean_correlation_matrix_rie_spectral,
)

_EIGENVALUE_FLOOR = 1e-10
_DENOMINATOR_FLOOR = 1e-12


@dataclass(frozen=True)
class CorrelationCleanerResult:
    method: str
    matrix_type: str
    sample_size: int | None
    input_matrix: pd.DataFrame
    empirical: RieSpectralDecomposition
    cleaned_eigenvalues_by_input_order: np.ndarray
    cleaned: RieSpectralDecomposition
    cleaned_matrix_pre_projection: pd.DataFrame
    cleaned_matrix: pd.DataFrame
    postprocess_applied: bool
    postprocess_steps: tuple[str, ...]
    post_projection: RieSpectralDecomposition | None = None
    spectral_source: str = "cleaned_matrix_diagonalization"


def _to_frame(matrix: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=index, columns=index)


def _to_eigenvector_frame(eigenvectors: np.ndarray, index: pd.Index) -> pd.DataFrame:
    columns = [f"rank_{rank}" for rank in range(1, eigenvectors.shape[1] + 1)]
    return pd.DataFrame(eigenvectors, index=index, columns=columns)


def _renormalize_to_correlation(matrix: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.clip(np.diag(matrix), _DENOMINATOR_FLOOR, None))
    corr = matrix / np.outer(diag, diag)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def _normalize_input_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    corr = make_psd(corr, floor=_EIGENVALUE_FLOOR)
    normalized = _renormalize_to_correlation(corr.to_numpy(dtype=float))
    return _to_frame(normalized, corr.index)


def _spectral_decomposition(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.to_numpy(dtype=float))
    order = tuple(int(idx) for idx in np.argsort(eigenvalues)[::-1])
    return eigenvalues.astype(float), eigenvectors.astype(float), order


def _build_decomposition(
    eigenvalues_by_input_order: np.ndarray,
    eigenvectors_by_input_order: np.ndarray,
    *,
    rank_order: tuple[int, ...],
    index: pd.Index,
) -> RieSpectralDecomposition:
    return RieSpectralDecomposition(
        eigenvalues=eigenvalues_by_input_order[list(rank_order)],
        eigenvectors=_to_eigenvector_frame(eigenvectors_by_input_order[:, list(rank_order)], index),
        rank_order=rank_order,
    )


def _rich_from_matrix(
    matrix: pd.DataFrame,
    *,
    method: str,
    sample_size: int | None,
    spectral_source: str,
) -> CorrelationCleanerResult:
    normalized = _normalize_input_correlation(matrix)
    empirical_vals, empirical_vecs, empirical_order = _spectral_decomposition(normalized)
    empirical = _build_decomposition(
        empirical_vals,
        empirical_vecs,
        rank_order=empirical_order,
        index=normalized.index,
    )
    return CorrelationCleanerResult(
        method=method,
        matrix_type="correlation",
        sample_size=sample_size,
        input_matrix=normalized,
        empirical=empirical,
        cleaned_eigenvalues_by_input_order=empirical_vals,
        cleaned=empirical,
        cleaned_matrix_pre_projection=normalized,
        cleaned_matrix=normalized,
        postprocess_applied=False,
        postprocess_steps=(),
        post_projection=None,
        spectral_source=spectral_source,
    )


def _empirical_result(corr: pd.DataFrame, *, sample_size: int | None) -> CorrelationCleanerResult:
    return _rich_from_matrix(
        corr,
        method="empirical",
        sample_size=sample_size,
        spectral_source="matrix_diagonalization",
    )


def _linear_shrinkage_result(
    corr: pd.DataFrame,
    *,
    sample_size: int | None,
    linear_shrinkage: float,
) -> CorrelationCleanerResult:
    normalized = _normalize_input_correlation(corr)
    empirical_vals, empirical_vecs, empirical_order = _spectral_decomposition(normalized)
    empirical = _build_decomposition(
        empirical_vals,
        empirical_vecs,
        rank_order=empirical_order,
        index=normalized.index,
    )
    shrinkage = float(np.clip(linear_shrinkage, 0.0, 1.0))
    cleaned_eigenvalues = ((1.0 - shrinkage) * empirical_vals) + shrinkage
    cleaned_matrix = _to_frame(
        empirical_vecs @ np.diag(cleaned_eigenvalues) @ empirical_vecs.T,
        normalized.index,
    )
    cleaned = _build_decomposition(
        cleaned_eigenvalues,
        empirical_vecs,
        rank_order=empirical_order,
        index=normalized.index,
    )
    return CorrelationCleanerResult(
        method="linear_shrinkage",
        matrix_type="correlation",
        sample_size=sample_size,
        input_matrix=normalized,
        empirical=empirical,
        cleaned_eigenvalues_by_input_order=cleaned_eigenvalues,
        cleaned=cleaned,
        cleaned_matrix_pre_projection=cleaned_matrix,
        cleaned_matrix=cleaned_matrix,
        postprocess_applied=False,
        postprocess_steps=(),
        post_projection=None,
        spectral_source="empirical_vectors_plus_affine_eigenvalue_shrinkage",
    )


def _rie_spectral_result(corr: pd.DataFrame, *, sample_size: int | None) -> CorrelationCleanerResult:
    if sample_size is None:
        raise ValueError("clean_correlation_matrix_rich(method='rie_spectral') requires sample_size.")
    result: RieCorrelationResult = clean_correlation_matrix_rie_spectral(corr, sample_size=int(sample_size))
    return CorrelationCleanerResult(
        method="rie_spectral",
        matrix_type=result.matrix_type,
        sample_size=result.sample_size,
        input_matrix=result.input_matrix,
        empirical=result.empirical,
        cleaned_eigenvalues_by_input_order=result.cleaned_eigenvalues_by_input_order,
        cleaned=result.cleaned,
        cleaned_matrix_pre_projection=result.cleaned_matrix_pre_projection,
        cleaned_matrix=result.cleaned_matrix,
        postprocess_applied=result.postprocess_applied,
        postprocess_steps=result.postprocess_steps,
        post_projection=result.post_projection,
        spectral_source=result.spectral_source,
    )


def _rie_reference_result(
    corr: pd.DataFrame,
    *,
    data: pd.DataFrame | None,
    sample_size: int | None,
    linear_shrinkage: float,
    bandwidth: float,
) -> CorrelationCleanerResult:
    normalized = _normalize_input_correlation(corr)
    empirical_vals, empirical_vecs, empirical_order = _spectral_decomposition(normalized)
    empirical = _build_decomposition(
        empirical_vals,
        empirical_vecs,
        rank_order=empirical_order,
        index=normalized.index,
    )
    cleaned_matrix = clean_correlation_matrix(
        normalized,
        data=data,
        sample_size=sample_size,
        method="rie_reference",
        linear_shrinkage=linear_shrinkage,
        bandwidth=bandwidth,
    )
    cleaned_vals, cleaned_vecs, cleaned_order = _spectral_decomposition(cleaned_matrix)
    cleaned = _build_decomposition(
        cleaned_vals,
        cleaned_vecs,
        rank_order=cleaned_order,
        index=cleaned_matrix.index,
    )
    return CorrelationCleanerResult(
        method="rie_reference",
        matrix_type="correlation",
        sample_size=sample_size,
        input_matrix=normalized,
        empirical=empirical,
        cleaned_eigenvalues_by_input_order=cleaned_vals,
        cleaned=cleaned,
        cleaned_matrix_pre_projection=cleaned_matrix,
        cleaned_matrix=cleaned_matrix,
        postprocess_applied=False,
        postprocess_steps=(),
        post_projection=None,
        spectral_source="cleaned_matrix_diagonalization",
    )


def clean_correlation_matrix_rich(
    corr: pd.DataFrame,
    *,
    data: pd.DataFrame | None = None,
    sample_size: int | None = None,
    method: str = "empirical",
    linear_shrinkage: float = 0.0,
    bandwidth: float = 1e-3,
) -> CorrelationCleanerResult:
    resolved_method = str(method).strip().lower()
    if resolved_method == "empirical":
        return _empirical_result(corr, sample_size=sample_size)
    if resolved_method == "linear_shrinkage":
        return _linear_shrinkage_result(
            corr,
            sample_size=sample_size,
            linear_shrinkage=linear_shrinkage,
        )
    if resolved_method in {"rie_spectral", "rie"}:
        return _rie_spectral_result(corr, sample_size=sample_size)
    if resolved_method == "rie_reference":
        return _rie_reference_result(
            corr,
            data=data,
            sample_size=sample_size,
            linear_shrinkage=linear_shrinkage,
            bandwidth=bandwidth,
        )
    raise ValueError(f"Unknown cleaning method '{method}'")


__all__ = [
    "CorrelationCleanerResult",
    "clean_correlation_matrix_rich",
]
