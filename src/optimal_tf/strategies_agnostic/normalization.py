from __future__ import annotations

import pandas as pd


def supported_normalization_modes() -> list[str]:
    """Return the exposed normalization modes for agnostic strategies."""
    return ["gross", "raw"]


def scale_to_target_l1(weights: pd.Series, *, target_l1: float = 1.0) -> pd.Series:
    """Scale a vector to a fixed L1 norm."""
    weights = weights.astype(float).fillna(0.0)
    gross = float(weights.abs().sum())
    if gross <= 0.0:
        return weights
    return weights * (float(target_l1) / gross)


def normalize_by_gross_exposure(weights: pd.Series) -> pd.Series:
    """Match the current project convention for long/short strategies."""
    return scale_to_target_l1(weights, target_l1=1.0)


def apply_long_only_projection(weights: pd.Series) -> pd.Series:
    """Project a vector into the long-only simplex."""
    clipped = weights.astype(float).fillna(0.0).clip(lower=0.0)
    total = float(clipped.sum())
    return clipped / total if total > 0.0 else clipped
