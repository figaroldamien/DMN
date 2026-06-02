"""Shared reporting, metrics, plots, and export helpers."""

from .benchmarks import equal_weight_buy_and_hold_benchmark, equal_weight_rebalanced_benchmark, single_asset_buy_and_hold_benchmark
from .exporters import write_evaluation_outputs
from .metrics import EvaluationSummary, Performance, evaluation_metrics, performance_metrics
from .plots import cumulative_nav, render_evaluation_plot, render_series_comparison_plot

__all__ = [
    "EvaluationSummary",
    "Performance",
    "cumulative_nav",
    "equal_weight_buy_and_hold_benchmark",
    "equal_weight_rebalanced_benchmark",
    "single_asset_buy_and_hold_benchmark",
    "evaluation_metrics",
    "performance_metrics",
    "render_evaluation_plot",
    "render_series_comparison_plot",
    "write_evaluation_outputs",
]
