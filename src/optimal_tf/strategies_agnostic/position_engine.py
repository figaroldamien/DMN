from __future__ import annotations

import numpy as np
import pandas as pd

from trading_core.risk import eigen_decomposition


def inverse_sqrt_operator(
    matrix: pd.DataFrame,
    *,
    floor: float = 1e-8,
) -> pd.DataFrame:
    """Return the symmetric inverse square-root of a PSD matrix."""
    vals, vecs = eigen_decomposition(matrix)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(vals, floor, None)))
    operator = vecs @ inv_sqrt @ vecs.T
    return pd.DataFrame(operator, index=matrix.index, columns=matrix.columns, dtype=float)


def build_agnostic_positions(
    corr: pd.DataFrame,
    q_matrix: pd.DataFrame,
    signal: pd.Series,
    *,
    omega: float = 1.0,
    floor: float = 1e-8,
) -> pd.Series:
    """Apply the Eq. 8-style position engine on aligned inputs.

    The engine computes:

        w = omega * C^{-1/2} * Q^{-1/2} * p

    where:
    - ``C`` is the cleaned correlation matrix
    - ``Q`` is the signal-covariance model
    - ``p`` is the current signal vector
    """
    assets = list(corr.index)
    aligned_q = q_matrix.reindex(index=assets, columns=assets).astype(float)
    aligned_signal = signal.reindex(assets).fillna(0.0).astype(float)

    c_inv_sqrt = inverse_sqrt_operator(corr, floor=floor)
    q_inv_sqrt = inverse_sqrt_operator(aligned_q, floor=floor)
    raw = float(omega) * (c_inv_sqrt.to_numpy(dtype=float) @ q_inv_sqrt.to_numpy(dtype=float) @ aligned_signal.to_numpy(dtype=float))
    return pd.Series(raw, index=assets, dtype=float)
