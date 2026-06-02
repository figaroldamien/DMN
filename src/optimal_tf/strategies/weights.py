from __future__ import annotations

"""Shared weight-normalization helpers used across strategy modules."""

import pandas as pd


def normalize_weights(weights: pd.Series, *, long_only: bool = False) -> pd.Series:
    """Normalize a raw weight vector to the project conventions.

    Two normalization modes are used in the strategy codebase:
    - ``long_only=True``: clip negatives and normalize by the simple sum
    - ``long_only=False``: keep signs and normalize by gross exposure

    This helper is intentionally tiny because it sits on the hot path of many
    strategy builders, but having it centralized guarantees that all strategies
    use the same leverage convention.
    """
    weights = weights.astype(float).fillna(0.0)
    if long_only:
        weights = weights.clip(lower=0.0)
        total = float(weights.sum())
        return weights / total if total > 0 else weights
    gross = float(weights.abs().sum())
    return weights / gross if gross > 0 else weights
