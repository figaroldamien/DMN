"""Shared helpers for optimal_tf research scripts."""

from .benchmark import (
    build_normalized_returns,
    eigenvalue_rows,
    matrix_benchmark_rows,
    matrix_sample,
    reference_pipe_row,
    render_scree_overview,
    strategy_benchmark_rows,
    write_matrix_pivots,
    write_strategy_pivots,
)
from .cli import (
    merge_common_overrides,
    parse_csv_list,
    parse_windows,
    resolve_target_dates,
    resolve_window_estimation_cfg,
    validate_methods,
    validate_strategies,
)

__all__ = [
    "build_normalized_returns",
    "eigenvalue_rows",
    "matrix_benchmark_rows",
    "matrix_sample",
    "merge_common_overrides",
    "parse_csv_list",
    "parse_windows",
    "reference_pipe_row",
    "render_scree_overview",
    "resolve_target_dates",
    "resolve_window_estimation_cfg",
    "strategy_benchmark_rows",
    "validate_methods",
    "validate_strategies",
    "write_matrix_pivots",
    "write_strategy_pivots",
    "build_scenario_highlights",
    "build_scenario_summary",
    "render_scenario_summary_text",
]

from .summary import build_scenario_highlights, build_scenario_summary, render_scenario_summary_text
