from __future__ import annotations

import pandas as pd

from .config import BacktestConfig, EstimationConfig
from trading_core.risk import supported_cleaning_methods


def compare_cleaners(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float]:
    # This stays intentionally simple: for the RIE step we mainly want a small,
    # deterministic regression harness against a trusted external implementation.
    diff = (reference - candidate).to_numpy(dtype=float)
    return {
        "max_abs_diff": float(abs(diff).max()),
        "mean_abs_diff": float(abs(diff).mean()),
    }


def validate_estimation_config(cfg: EstimationConfig) -> None:
    if cfg.cleaning_method not in supported_cleaning_methods():
        raise ValueError(
            f"cleaning_method must be one of {list(supported_cleaning_methods())} "
            f"(got {cfg.cleaning_method!r})."
        )
    if cfg.covariance_window is not None and cfg.covariance_window <= 0:
        raise ValueError("covariance_window must be strictly positive.")
    if cfg.covariance_min_periods <= 0:
        raise ValueError("covariance_min_periods must be strictly positive.")
    if cfg.covariance_window is not None and cfg.covariance_min_periods > cfg.covariance_window:
        raise ValueError(
            "covariance_min_periods must be less than or equal to covariance_window "
            f"(got covariance_min_periods={cfg.covariance_min_periods}, covariance_window={cfg.covariance_window})."
        )


def validate_backtest_config(cfg: BacktestConfig) -> None:
    if not 0.0 < cfg.weight_smoothing_alpha <= 1.0:
        raise ValueError(
            "weight_smoothing_alpha must be in the interval (0, 1] "
            f"(got weight_smoothing_alpha={cfg.weight_smoothing_alpha})."
        )
