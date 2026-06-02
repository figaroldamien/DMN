from __future__ import annotations

"""Naive Markowitz strategy.

This module implements the simplest possible mean-variance allocation with a
flat expected-return assumption.

Why "naive":
- we do not estimate asset-specific expected returns here
- we simply assume every asset has the same expected excess return
- all cross-sectional differentiation therefore comes from the covariance
  structure alone

This makes NM a useful diagnostic strategy because it reveals what the inverse
covariance matrix alone implies for relative weights.
"""

import numpy as np
import pandas as pd

from .weights import normalize_weights


def naive_markowitz_weights_from_cov(cov: pd.DataFrame) -> pd.Series:
    """Solve a flat-mean Markowitz allocation.

    With expected returns assumed constant across assets, the classical
    mean-variance optimizer reduces to something proportional to:

        w ~ Sigma^{-1} 1

    where:
    - ``Sigma`` is the covariance matrix
    - ``1`` is the vector of identical expected returns

    We use the pseudo-inverse instead of the exact inverse so the routine stays
    numerically robust even when the covariance matrix is ill-conditioned or
    close to singular.
    """
    cov_arr = cov.to_numpy(dtype=float)

    # "Naive Markowitz" uses a flat expected return vector and leaves the
    # covariance matrix to determine the relative allocations.
    raw = np.linalg.pinv(cov_arr) @ np.ones(cov_arr.shape[0], dtype=float)

    # Normalize to unit gross exposure so the result fits the same conventions
    # as the other cross-sectional strategy builders.
    return normalize_weights(pd.Series(raw, index=cov.index, dtype=float), long_only=False)
