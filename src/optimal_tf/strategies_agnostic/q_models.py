from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from trading_core.risk import clean_correlation_matrix, make_psd

from ..config import EstimationConfig

QModel = Literal["identity", "correlation", "phi_shrink_correlation", "empirical"]
QMatrixKind = Literal["structural", "empirical"]


def supported_q_models() -> list[str]:
    """Return the dashboard/service-visible agnostic Q-model identifiers."""
    return ["identity", "correlation", "phi_shrink_correlation", "empirical"]


def identity_q_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Return the agnostic ``Q = I`` benchmark."""
    return pd.DataFrame(np.eye(len(corr), dtype=float), index=corr.index, columns=corr.columns)


def correlation_q_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Return ``Q = C``."""
    return corr.astype(float).copy()


def phi_shrink_correlation_q_matrix(
    corr: pd.DataFrame,
    *,
    phi: float,
) -> pd.DataFrame:
    """Return ``Q_phi = phi * C + (1 - phi) * I``."""
    if not 0.0 <= float(phi) <= 1.0:
        raise ValueError(f"phi must lie in [0, 1], got {phi}.")
    ident = identity_q_matrix(corr)
    return (float(phi) * corr.astype(float)) + ((1.0 - float(phi)) * ident)


def build_q_matrix(
    corr: pd.DataFrame,
    *,
    q_model: QModel,
    phi: float,
    signal_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the raw Q matrix for one named agnostic model."""
    if q_model == "identity":
        return identity_q_matrix(corr)
    if q_model == "correlation":
        return correlation_q_matrix(corr)
    if q_model == "phi_shrink_correlation":
        return phi_shrink_correlation_q_matrix(corr, phi=phi)
    if q_model == "empirical":
        if signal_panel is None:
            raise ValueError("q_model='empirical' requires a signal_panel.")
        return empirical_signal_q_matrix(signal_panel, assets=corr.index)
    raise ValueError(f"Unknown q_model '{q_model}'. Allowed values: {supported_q_models()}.")


def empirical_signal_q_matrix(
    signal_panel: pd.DataFrame,
    *,
    assets: pd.Index | list[str],
) -> pd.DataFrame:
    """Estimate a raw empirical signal-correlation matrix from signal history."""
    asset_index = pd.Index(assets)
    sample = signal_panel.reindex(columns=asset_index).astype(float)
    # Constant or empty columns cannot define a meaningful empirical signal
    # correlation. We keep only columns with variation and enough data.
    valid = sample.columns[(sample.notna().sum(axis=0) >= 2) & (sample.std(axis=0, skipna=True) > 0.0)]
    if len(valid) < 2:
        raise ValueError("Empirical Q requires at least two signal series with variation.")
    corr = sample.loc[:, valid].corr(min_periods=2)
    keep = corr.index[~corr.isna().any(axis=1)]
    corr = corr.loc[keep, keep]
    if corr.empty or len(corr) < 2:
        raise ValueError("Empirical Q could not be estimated from the available signal history.")
    return corr.astype(float)


def clean_empirical_matrix(
    matrix: pd.DataFrame,
    *,
    est_cfg: EstimationConfig,
    data: pd.DataFrame | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Apply the configured statistical cleaner to one empirical matrix."""
    return clean_correlation_matrix(
        matrix,
        data=data,
        sample_size=sample_size,
        method=est_cfg.cleaning_method,
        linear_shrinkage=est_cfg.linear_shrinkage,
        bandwidth=est_cfg.rie_bandwidth,
    )


def clean_structural_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit structural repair to a closed-form Q matrix.

    The algorithm is intentionally local and explicit:
    - symmetrize the matrix
    - project it to PSD
    - renormalize it back to a correlation matrix
    """
    symmetric = 0.5 * (matrix.astype(float) + matrix.astype(float).T)
    psd = make_psd(symmetric)
    diag = np.sqrt(np.clip(np.diag(psd.to_numpy(dtype=float)), 1e-12, None))
    renormalized = psd.to_numpy(dtype=float) / np.outer(diag, diag)
    renormalized = 0.5 * (renormalized + renormalized.T)
    np.fill_diagonal(renormalized, 1.0)
    return pd.DataFrame(renormalized, index=matrix.index, columns=matrix.columns, dtype=float)


def q_matrix_kind(q_model: QModel) -> QMatrixKind:
    """Classify the current Q builder by how its output should be cleaned."""
    if q_model in {"identity", "correlation", "phi_shrink_correlation"}:
        return "structural"
    if q_model == "empirical":
        return "empirical"
    raise ValueError(f"Unknown q_model '{q_model}'. Allowed values: {supported_q_models()}.")


def clean_q_matrix(
    q_matrix: pd.DataFrame,
    *,
    q_kind: QMatrixKind,
    est_cfg: EstimationConfig,
    data: pd.DataFrame | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Dispatch Q cleaning according to whether the matrix is structural or empirical."""
    if q_kind == "structural":
        return clean_structural_matrix(q_matrix)
    if q_kind == "empirical":
        return clean_empirical_matrix(q_matrix, est_cfg=est_cfg, data=data, sample_size=sample_size)
    raise ValueError(f"Unknown q_kind '{q_kind}'.")


def resolve_q_matrix(
    corr: pd.DataFrame,
    *,
    q_model: QModel,
    phi: float,
    est_cfg: EstimationConfig,
    signal_panel: pd.DataFrame | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Build and clean Q behind one model-level interface."""
    raw = build_q_matrix(corr, q_model=q_model, phi=phi, signal_panel=signal_panel)
    return clean_q_matrix(
        raw,
        q_kind=q_matrix_kind(q_model),
        est_cfg=est_cfg,
        data=signal_panel,
        sample_size=sample_size,
    )
