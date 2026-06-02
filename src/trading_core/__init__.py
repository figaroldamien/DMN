"""Shared trading research primitives used by application packages."""

from .backtest import ComparisonResult, EvaluationResult, compare_strategies, evaluate_portfolio
from .data import load_prices_for_universe, load_prices_yf
from .features import (
    compute_returns,
    effective_span_from_alpha,
    ewma_cov_frame,
    ewma_vol,
    normalize_returns_by_vol,
    sanitize_returns,
    trend_ema_signal,
)
from .market import get_universe_tickers, list_universes
from .rebalance import resolve_rebalance_dates, supported_rebalance_frequencies
from .reporting import (
    EvaluationSummary,
    cumulative_nav,
    equal_weight_buy_and_hold_benchmark,
    equal_weight_rebalanced_benchmark,
    single_asset_buy_and_hold_benchmark,
    render_evaluation_plot,
    render_series_comparison_plot,
    write_evaluation_outputs,
)
from .risk import (
    clean_correlation_matrix,
    correlation_to_covariance,
    covariance_to_correlation,
    eigen_decomposition,
    estimate_clean_covariance_at_date,
    estimate_clean_covariance_panel,
    make_psd,
)

__all__ = [
    "ComparisonResult",
    "EvaluationResult",
    "EvaluationSummary",
    "clean_correlation_matrix",
    "compare_strategies",
    "compute_returns",
    "correlation_to_covariance",
    "covariance_to_correlation",
    "cumulative_nav",
    "effective_span_from_alpha",
    "eigen_decomposition",
    "equal_weight_buy_and_hold_benchmark",
    "equal_weight_rebalanced_benchmark",
    "single_asset_buy_and_hold_benchmark",
    "estimate_clean_covariance_at_date",
    "estimate_clean_covariance_panel",
    "evaluate_portfolio",
    "ewma_cov_frame",
    "ewma_vol",
    "get_universe_tickers",
    "list_universes",
    "load_prices_for_universe",
    "load_prices_yf",
    "make_psd",
    "normalize_returns_by_vol",
    "render_evaluation_plot",
    "render_series_comparison_plot",
    "resolve_rebalance_dates",
    "sanitize_returns",
    "supported_rebalance_frequencies",
    "trend_ema_signal",
    "write_evaluation_outputs",
]
