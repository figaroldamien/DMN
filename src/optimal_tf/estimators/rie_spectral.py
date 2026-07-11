from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_core.risk.covariance import correlation_to_covariance, covariance_to_correlation

_EIGENVALUE_FLOOR = 1e-12
_DENOMINATOR_FLOOR = 1e-12


@dataclass(frozen=True)
class RieSpectralDecomposition:
    eigenvalues: np.ndarray
    eigenvectors: pd.DataFrame
    rank_order: tuple[int, ...]


@dataclass(frozen=True)
class RieCorrelationResult:
    matrix_type: str
    sample_size: int
    input_matrix: pd.DataFrame
    empirical: RieSpectralDecomposition
    cleaned_eigenvalues_by_input_order: np.ndarray
    cleaned: RieSpectralDecomposition
    cleaned_matrix_pre_projection: pd.DataFrame
    cleaned_matrix: pd.DataFrame
    postprocess_applied: bool
    postprocess_steps: tuple[str, ...]
    post_projection: RieSpectralDecomposition | None = None
    spectral_source: str = "empirical_vectors_plus_xi_hat"


@dataclass(frozen=True)
class RieCovarianceResult:
    matrix_type: str
    sample_size: int
    input_matrix: pd.DataFrame
    volatility: pd.Series
    correlation_result: RieCorrelationResult
    cleaned_matrix_pre_projection: pd.DataFrame
    cleaned_matrix: pd.DataFrame


def _to_frame(matrix: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=index, columns=index)


def _to_eigenvector_frame(eigenvectors: np.ndarray, index: pd.Index) -> pd.DataFrame:
    columns = [f"rank_{rank}" for rank in range(1, eigenvectors.shape[1] + 1)]
    return pd.DataFrame(eigenvectors, index=index, columns=columns)


def _symmetric_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    arr = matrix.to_numpy(dtype=float)
    arr = 0.5 * (arr + arr.T)
    return _to_frame(arr, matrix.index)


def _renormalize_to_correlation(matrix: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.clip(np.diag(matrix), _DENOMINATOR_FLOOR, None))
    corr = matrix / np.outer(diag, diag)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def _validate_square_frame(matrix: pd.DataFrame, *, matrix_name: str) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError(f"{matrix_name} must be a pandas DataFrame.")
    if matrix.empty:
        raise ValueError(f"{matrix_name} must not be empty.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{matrix_name} must be square.")
    if not matrix.index.equals(matrix.columns):
        raise ValueError(f"{matrix_name} must have identical row/column labels.")
    return _symmetric_frame(matrix.astype(float))


def _empirical_spectral_decomposition(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.to_numpy(dtype=float))
    return eigenvalues.astype(float), eigenvectors.astype(float)


def _compute_xi_hat(empirical_eigenvalues: np.ndarray, *, sample_size: int) -> np.ndarray:
    num_assets = len(empirical_eigenvalues)
    if num_assets < 2:
        raise ValueError("RIE requires at least two assets.")
    if sample_size <= 0:
        raise ValueError("sample_size must be strictly positive.")

    stable_eigenvalues = np.clip(empirical_eigenvalues.astype(float), _EIGENVALUE_FLOOR, None)
    q = float(num_assets / sample_size)
    n_lambda = stable_eigenvalues[0]
    sigma_sq = n_lambda / (1.0 - np.sqrt(q)) ** 2
    lambda_plus = n_lambda * ((1.0 + np.sqrt(q)) / (1.0 - np.sqrt(q))) ** 2
    z_k = stable_eigenvalues - (1j / np.sqrt(num_assets))

    s_k = np.empty(num_assets, dtype=np.complex128)
    for index_lambda in range(num_assets):
        inv_terms = 1.0 / (z_k[index_lambda] - stable_eigenvalues)
        inv_terms[index_lambda] = 0.0
        s_k[index_lambda] = np.sum(inv_terms) / num_assets

    xi_k = stable_eigenvalues / np.abs(1.0 - q + q * z_k * s_k) ** 2
    g_mp = (
        z_k
        + sigma_sq * (q - 1.0)
        - (np.sqrt(z_k - n_lambda) * np.sqrt(z_k - lambda_plus))
    ) / (2.0 * q * z_k * sigma_sq)
    gamma_k = sigma_sq * (np.abs(1.0 - q + q * z_k * g_mp) ** 2 / stable_eigenvalues)
    xi_hat = np.array(
        [xi * gamma if gamma > 1.0 else xi for xi, gamma in zip(xi_k, gamma_k, strict=True)],
        dtype=float,
    )
    return np.clip(xi_hat, _EIGENVALUE_FLOOR, None)


def _postprocess_correlation_matrix(
    matrix: pd.DataFrame,
    *,
    force_unit_diagonal: bool,
    clip_upper_bound: float | None,
    renormalize_output: bool,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    arr = matrix.to_numpy(dtype=float).copy()
    steps: list[str] = []
    if force_unit_diagonal:
        np.fill_diagonal(arr, 1.0)
        steps.append("force_unit_diagonal")
    if clip_upper_bound is not None:
        arr[arr > float(clip_upper_bound)] = float(clip_upper_bound)
        steps.append(f"clip_upper_bound={float(clip_upper_bound):g}")
    if renormalize_output:
        arr = _renormalize_to_correlation(arr)
        steps.append("renormalize_to_correlation")
    return _to_frame(arr, matrix.index), tuple(steps)


def clean_correlation_matrix_rie_spectral(
    corr: pd.DataFrame,
    *,
    sample_size: int,
    force_unit_diagonal: bool = True,
    clip_upper_bound: float | None = 1.0,
    renormalize_output: bool = True,
) -> RieCorrelationResult:
    corr = _validate_square_frame(corr, matrix_name="corr")
    empirical_eigenvalues, empirical_eigenvectors = _empirical_spectral_decomposition(corr)
    xi_hat = _compute_xi_hat(empirical_eigenvalues, sample_size=sample_size)

    cleaned_pre_projection_arr = empirical_eigenvectors @ np.diag(xi_hat) @ empirical_eigenvectors.T
    cleaned_pre_projection = _to_frame(cleaned_pre_projection_arr, corr.index)

    cleaned_order = tuple(int(idx) for idx in np.argsort(xi_hat)[::-1])
    cleaned_eigenvalues = xi_hat[list(cleaned_order)]
    cleaned_eigenvectors = empirical_eigenvectors[:, list(cleaned_order)]

    cleaned_matrix, postprocess_steps = _postprocess_correlation_matrix(
        cleaned_pre_projection,
        force_unit_diagonal=force_unit_diagonal,
        clip_upper_bound=clip_upper_bound,
        renormalize_output=renormalize_output,
    )

    post_projection = None
    if postprocess_steps:
        post_vals, post_vecs = _empirical_spectral_decomposition(cleaned_matrix)
        post_order = tuple(int(idx) for idx in np.argsort(post_vals)[::-1])
        post_projection = RieSpectralDecomposition(
            eigenvalues=post_vals[list(post_order)],
            eigenvectors=_to_eigenvector_frame(post_vecs[:, list(post_order)], corr.index),
            rank_order=post_order,
        )

    empirical_order = tuple(int(idx) for idx in np.argsort(empirical_eigenvalues)[::-1])
    empirical = RieSpectralDecomposition(
        eigenvalues=empirical_eigenvalues[list(empirical_order)],
        eigenvectors=_to_eigenvector_frame(empirical_eigenvectors[:, list(empirical_order)], corr.index),
        rank_order=empirical_order,
    )
    cleaned = RieSpectralDecomposition(
        eigenvalues=cleaned_eigenvalues,
        eigenvectors=_to_eigenvector_frame(cleaned_eigenvectors, corr.index),
        rank_order=cleaned_order,
    )

    return RieCorrelationResult(
        matrix_type="correlation",
        sample_size=int(sample_size),
        input_matrix=corr,
        empirical=empirical,
        cleaned_eigenvalues_by_input_order=xi_hat,
        cleaned=cleaned,
        cleaned_matrix_pre_projection=cleaned_pre_projection,
        cleaned_matrix=cleaned_matrix,
        postprocess_applied=bool(postprocess_steps),
        postprocess_steps=postprocess_steps,
        post_projection=post_projection,
    )


def clean_covariance_matrix_rie_spectral(
    cov: pd.DataFrame,
    *,
    sample_size: int,
    volatility: pd.Series | None = None,
    force_unit_diagonal: bool = True,
    clip_upper_bound: float | None = 1.0,
    renormalize_output: bool = True,
) -> RieCovarianceResult:
    cov = _validate_square_frame(cov, matrix_name="cov")
    if volatility is None:
        volatility = pd.Series(
            np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), _DENOMINATOR_FLOOR, None)),
            index=cov.index,
            dtype=float,
        )
    else:
        volatility = pd.Series(volatility, copy=True).reindex(cov.index).astype(float)
    corr = covariance_to_correlation(cov)
    corr_result = clean_correlation_matrix_rie_spectral(
        corr,
        sample_size=sample_size,
        force_unit_diagonal=force_unit_diagonal,
        clip_upper_bound=clip_upper_bound,
        renormalize_output=renormalize_output,
    )
    cleaned_pre_projection_cov = correlation_to_covariance(corr_result.cleaned_matrix_pre_projection, volatility)
    cleaned_cov = correlation_to_covariance(corr_result.cleaned_matrix, volatility)
    return RieCovarianceResult(
        matrix_type="covariance",
        sample_size=int(sample_size),
        input_matrix=cov,
        volatility=volatility,
        correlation_result=corr_result,
        cleaned_matrix_pre_projection=cleaned_pre_projection_cov,
        cleaned_matrix=cleaned_cov,
    )


def clean_matrix_rie_spectral(
    matrix: pd.DataFrame,
    *,
    matrix_type: str,
    sample_size: int,
    volatility: pd.Series | None = None,
    force_unit_diagonal: bool = True,
    clip_upper_bound: float | None = 1.0,
    renormalize_output: bool = True,
) -> RieCorrelationResult | RieCovarianceResult:
    resolved_type = str(matrix_type).strip().lower()
    if resolved_type == "correlation":
        return clean_correlation_matrix_rie_spectral(
            matrix,
            sample_size=sample_size,
            force_unit_diagonal=force_unit_diagonal,
            clip_upper_bound=clip_upper_bound,
            renormalize_output=renormalize_output,
        )
    if resolved_type == "covariance":
        return clean_covariance_matrix_rie_spectral(
            matrix,
            sample_size=sample_size,
            volatility=volatility,
            force_unit_diagonal=force_unit_diagonal,
            clip_upper_bound=clip_upper_bound,
            renormalize_output=renormalize_output,
        )
    raise ValueError("matrix_type must be either 'correlation' or 'covariance'.")


__all__ = [
    "RieCorrelationResult",
    "RieCovarianceResult",
    "RieSpectralDecomposition",
    "clean_correlation_matrix_rie_spectral",
    "clean_covariance_matrix_rie_spectral",
    "clean_matrix_rie_spectral",
]
