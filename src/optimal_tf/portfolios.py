"""Compatibility facade for portfolio helpers.

Historical imports still reference `optimal_tf.portfolios`. The actual shared
helpers now live in `optimal_tf.portfolio_helpers`.
"""

from .portfolio_helpers import (
    agnostic_risk_parity_weights_from_cov,
    naive_markowitz_weights_from_cov,
    normalize_weights,
    risk_parity_weights_from_cov,
)

__all__ = [
    "agnostic_risk_parity_weights_from_cov",
    "naive_markowitz_weights_from_cov",
    "normalize_weights",
    "risk_parity_weights_from_cov",
]
