from __future__ import annotations

"""Risk Parity baseline strategy.

This module intentionally keeps the implementation very small, but the comments
spell out the financial interpretation in detail because RP is the reference
portfolio used by several higher-level strategies in the project.

Current design choice:
- this is *not* a full equal-risk-contribution optimizer yet
- instead, we use inverse-volatility weights as a simple and robust proxy

Why this still deserves to live in its own module:
- RP is a first-class strategy in the research framework
- having a dedicated file makes it easier to later swap the implementation for
  a true ERC solver without touching unrelated strategy code
"""

import numpy as np
import pandas as pd


def risk_parity_weights_from_cov(cov: pd.DataFrame) -> pd.Series:
    """Build a simple Risk Parity allocation from a covariance matrix.

    Financial intuition:
    - assets with high variance should receive less capital
    - assets with low variance should receive more capital
    - with this approximation, risk budgeting is enforced through marginal vol
      only, not through a full contribution-to-risk fixed-point solver

    Mathematical approximation used here:
    - compute each asset volatility from the covariance diagonal
    - assign raw weights proportional to ``1 / vol_i``
    - normalize the final vector to unit gross exposure

    This is often called an "inverse-vol" portfolio. It is weaker than a true
    equal-risk-contribution solution, but it is fast, stable, and easy to audit,
    which makes it a good baseline for research and benchmarking.
    """
    # The covariance diagonal is the asset variance. We clip it to avoid taking
    # square roots of tiny negative values coming from floating-point noise.
    vol = pd.Series(
        np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), 1e-12, None)),
        index=cov.index,
    )

    # Inverse-vol weighting penalizes already risky assets and increases the
    # capital allocated to quieter assets. This is the simplest RP proxy.
    weights = 1.0 / vol

    # We normalize by gross exposure so the absolute weights sum to one even if
    # some future variant introduces signed positions.
    return weights / weights.abs().sum()
