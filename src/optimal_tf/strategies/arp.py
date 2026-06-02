from __future__ import annotations

"""Agnostic Risk Parity strategy.

ARP starts from a different intuition than plain Risk Parity.

Instead of allocating capital directly in asset space from variances only, it:
- converts the covariance matrix to a correlation matrix
- diagonalizes that correlation matrix into orthogonal risk modes
- whitens those modes so each decorrelated component carries the same ex-ante
  budget
- maps the result back into asset space

The goal is to avoid over-allocating to a small number of strongly correlated
assets that would look diversified in raw weight space but not in factor space.
"""

import numpy as np
import pandas as pd

from trading_core.risk import covariance_to_correlation, eigen_decomposition


def agnostic_risk_parity_weights_from_cov(cov: pd.DataFrame) -> pd.Series:
    """Construct ARP weights from a covariance matrix.

    High-level derivation:
    1. Convert covariance to correlation so the spectrum reflects dependency
       structure independently from volatility scale.
    2. Eigendecompose the correlation matrix into orthogonal eigen-portfolios.
    3. Divide each mode by the square root of its eigenvalue.
       This is the whitening step: large common factors get shrunk, while weak
       modes are brought to the same ex-ante variance scale.
    4. Recombine the transformed modes back into asset space.
    5. Normalize the resulting allocation by gross exposure.

    The final vector can be seen as the portfolio that gives equal importance to
    decorrelated correlation modes rather than to the original asset axes.
    """
    # ARP works on correlation rather than covariance because the strategy is
    # about redistributing exposure across dependency modes, not about reusing
    # the original per-asset volatility scale.
    corr = covariance_to_correlation(cov)

    # ``vals`` are the variances of the orthogonal correlation modes and
    # ``vecs`` gives the basis that rotates us from factor space back to assets.
    vals, vecs = eigen_decomposition(corr)

    # Whitening the correlation spectrum gives each decorrelated mode the same
    # ex-ante risk budget, which is the core intuition behind ARP.
    inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(vals, 1e-8, None)))

    # ``whitened`` is the linear operator that equalizes the variance of each
    # mode before projecting the result back into asset space.
    whitened = vecs @ inv_sqrt @ vecs.T

    # Applying the whitening operator to an all-ones vector yields a symmetric
    # "mode-balanced" portfolio in asset coordinates.
    ones = np.ones(len(corr))
    weights = whitened @ ones
    series = pd.Series(weights, index=corr.index, dtype=float)

    # Gross normalization keeps the portfolio comparable to the other strategy
    # implementations in the codebase.
    return series / series.abs().sum()
