"""Shared return, volatility, trend, and panel transformation helpers."""

from .returns import compute_returns, rolling_return, sanitize_returns
from .transforms import normalize_returns_by_vol
from .trend import ema, macd, trend_ema_signal
from .volatility import alpha_from_span, effective_span_from_alpha, ewma_cov_frame, ewma_vol, resolve_ewma_alpha

__all__ = [
    "alpha_from_span",
    "compute_returns",
    "effective_span_from_alpha",
    "ema",
    "ewma_cov_frame",
    "ewma_vol",
    "macd",
    "normalize_returns_by_vol",
    "resolve_ewma_alpha",
    "rolling_return",
    "sanitize_returns",
    "trend_ema_signal",
]

