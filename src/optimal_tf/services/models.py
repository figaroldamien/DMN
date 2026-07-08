from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from trading_core.backtest.types import EvaluationResult


@dataclass(frozen=True)
class RunArtifacts:
    root_dir: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class AllocationRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    as_of_date: str | None = None
    strategy: str | None = None
    cleaning_method: str | None = None
    linear_shrinkage: float | None = None
    covariance_window: int | None = None
    long_only: bool | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None


@dataclass(frozen=True)
class AllocationResult:
    request: AllocationRequest
    universe: str
    strategy: str
    cleaning_method: str
    covariance_window: int | None
    allocation_date: pd.Timestamp
    signal_scale: float
    weights: pd.Series
    base_weights: pd.Series
    artifacts: RunArtifacts


@dataclass(frozen=True)
class StandardEvaluationRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    strategy: str | None = None
    cleaning_method: str | None = None
    linear_shrinkage: float | None = None
    covariance_window: int | None = None
    rebalance_frequency: str | None = None
    weight_smoothing_alpha: float | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    long_only: bool | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    output_plot: bool = True


@dataclass(frozen=True)
class StandardEvaluationResult:
    request: StandardEvaluationRequest
    universe: str
    strategy: str
    cleaning_method: str
    covariance_window: int | None
    rebalance_frequency: str
    evaluation_result: EvaluationResult
    benchmark_returns: pd.Series
    benchmark_label: str
    benchmark_metadata: dict[str, Any] | None
    buy_hold_returns: pd.Series
    buy_hold_label: str
    artifacts: RunArtifacts


@dataclass(frozen=True)
class StrategyTestbedRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    strategy: str | None = None
    cleaning_method: str | None = None
    linear_shrinkage: float | None = None
    covariance_window: int | None = None
    trend_alpha: float | None = None
    trend_span: int | None = None
    rebalance_frequency: str | None = None
    weight_smoothing_alpha: float | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    long_only: bool | None = None
    signal_model: str = "ones"
    q_model: str = "identity"
    phi: float = 0.0
    omega: float = 1.0
    normalization: str = "gross"
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    output_plot: bool = True


@dataclass(frozen=True)
class StrategyTestbedResult:
    request: StrategyTestbedRequest
    universe: str
    strategy_label: str
    cleaning_method: str
    covariance_window: int | None
    rebalance_frequency: str
    signal_model: str
    q_model: str
    phi: float
    omega: float
    normalization: str
    evaluation_result: EvaluationResult
    benchmark_returns: pd.Series
    benchmark_label: str
    benchmark_metadata: dict[str, Any] | None
    buy_hold_returns: pd.Series
    buy_hold_label: str
    artifacts: RunArtifacts


@dataclass(frozen=True)
class CompareRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    strategies: list[str] = field(default_factory=list)
    cleaning_method: str | None = None
    linear_shrinkage: float | None = None
    covariance_window: int | None = None
    rebalance_frequency: str | None = None
    weight_smoothing_alpha: float | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    long_only: bool | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    clean_output_dir: bool = True
    output_plot: bool = True


@dataclass(frozen=True)
class CompareResult:
    request: CompareRequest
    universe: str
    strategies: list[str]
    cleaning_method: str
    covariance_window: int | None
    rebalance_frequency: str
    comparison: Any
    benchmark_label: str
    benchmark_metadata: dict[str, Any] | None
    benchmark_nav: pd.Series
    benchmark_drawdown: pd.Series
    artifacts: RunArtifacts


@dataclass(frozen=True)
class MarketSynthesisRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    as_of_date: str | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None


@dataclass(frozen=True)
class MarketSynthesisResult:
    request: MarketSynthesisRequest
    universe: str
    start: str
    as_of_date: pd.Timestamp
    synthesis_mode: str
    has_hierarchy: bool
    consolidated_frame: pd.DataFrame
    ticker_frame: pd.DataFrame
    sector_nav_frame: pd.DataFrame
    ticker_nav_frame: pd.DataFrame
    monthly_consolidated_frame: pd.DataFrame
    monthly_ticker_frame: pd.DataFrame
    artifacts: RunArtifacts


@dataclass(frozen=True)
class VaryCleaningRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    rebalance_frequency: str | None = None
    strategy: str | None = None
    methods: list[str] = field(default_factory=list)
    linear_shrinkage: float | None = None
    window: int | None = None
    weight_smoothing_alpha: float | None = None
    matrix_date: str | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    log_scale: bool = True


@dataclass(frozen=True)
class VaryWindowRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    rebalance_frequency: str | None = None
    strategy: str | None = None
    method: str | None = None
    linear_shrinkage: float | None = None
    windows: list[int] = field(default_factory=list)
    weight_smoothing_alpha: float | None = None
    matrix_date: str | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    min_periods_mode: str = "clamp"
    log_scale: bool = True


@dataclass(frozen=True)
class VaryStrategyRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    rebalance_frequency: str | None = None
    strategies: list[str] = field(default_factory=list)
    method: str | None = None
    linear_shrinkage: float | None = None
    window: int | None = None
    weight_smoothing_alpha: float | None = None
    matrix_date: str | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    min_periods_mode: str = "clamp"


@dataclass(frozen=True)
class VaryFrequencyRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    strategy: str | None = None
    method: str | None = None
    linear_shrinkage: float | None = None
    window: int | None = None
    frequencies: list[str] = field(default_factory=list)
    weight_smoothing_alpha: float | None = None
    matrix_date: str | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    min_periods_mode: str = "clamp"


@dataclass(frozen=True)
class SpectrumByCleanerRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    rebalance_frequency: str | None = None
    methods: list[str] = field(default_factory=list)
    linear_shrinkage: float | None = None
    matrix_date: str | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    log_scale: bool = True


@dataclass(frozen=True)
class SpectrumByWindowRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    rebalance_frequency: str | None = None
    method: str | None = None
    linear_shrinkage: float | None = None
    windows: list[int] = field(default_factory=list)
    matrix_date: str | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    min_periods_mode: str = "clamp"
    log_scale: bool = True


@dataclass(frozen=True)
class EigenvectorInspectionRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    rebalance_frequency: str | None = None
    method: str = "rie_reference"
    linear_shrinkage: float | None = None
    windows: list[int] = field(default_factory=list)
    matrix_date: str | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    min_periods_mode: str = "clamp"
    selection_mode: str = "mp"
    selection_cumulative_variance: float = 80.0
    selection_top_n: int = 3
    log_scale: bool = True


@dataclass(frozen=True)
class ScenarioEvaluationResult:
    universe: str
    scenario_key: str
    scenario_summary: pd.DataFrame
    strategy_benchmark: pd.DataFrame
    matrix_benchmark: pd.DataFrame
    nav_comparison: pd.DataFrame
    drawdown_comparison: pd.DataFrame
    benchmark_label: str | None
    benchmark_summary: dict[str, Any] | None
    benchmark_nav: pd.Series
    benchmark_drawdown: pd.Series
    highlights: dict[str, str]
    artifacts: RunArtifacts


@dataclass(frozen=True)
class VaryCleaningResult(ScenarioEvaluationResult):
    request: VaryCleaningRequest
    scree_frame: pd.DataFrame


@dataclass(frozen=True)
class VaryWindowResult(ScenarioEvaluationResult):
    request: VaryWindowRequest
    scree_frame: pd.DataFrame


@dataclass(frozen=True)
class VaryStrategyResult(ScenarioEvaluationResult):
    request: VaryStrategyRequest


@dataclass(frozen=True)
class VaryFrequencyResult(ScenarioEvaluationResult):
    request: VaryFrequencyRequest


@dataclass(frozen=True)
class HyperparameterTuningRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    rebalance_frequency: str | None = None
    frequencies: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    linear_shrinkage: float | None = None
    windows: list[int] = field(default_factory=list)
    weight_smoothing_alpha: float | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None
    min_periods_mode: str = "clamp"


@dataclass(frozen=True)
class HyperparameterTuningResult:
    request: HyperparameterTuningRequest
    universe: str
    results_table: pd.DataFrame
    skipped_configs: pd.DataFrame
    highlights: dict[str, str]
    artifacts: RunArtifacts


@dataclass(frozen=True)
class SpectrumByCleanerResult:
    request: SpectrumByCleanerRequest
    universe: str
    matrix_date: pd.Timestamp
    methods: list[str]
    eigenvalue_frame: pd.DataFrame
    num_assets: int
    sample_size: int
    artifacts: RunArtifacts


@dataclass(frozen=True)
class SpectrumByWindowResult:
    request: SpectrumByWindowRequest
    universe: str
    matrix_date: pd.Timestamp
    method: str
    windows: list[int]
    scree_frame: pd.DataFrame
    artifacts: RunArtifacts


@dataclass(frozen=True)
class EigenvectorInspectionResult:
    request: EigenvectorInspectionRequest
    universe: str
    matrix_date: pd.Timestamp
    method: str
    windows: list[int]
    scree_frame: pd.DataFrame
    sector_presence: pd.DataFrame
    sector_signed: pd.DataFrame
    sub_sector_presence: pd.DataFrame
    sub_sector_signed: pd.DataFrame
    loadings: pd.DataFrame
    artifacts: RunArtifacts


@dataclass(frozen=True)
class InspectionSnapshotRequest:
    config_path: str = "configs/optimal_tf.example.toml"
    universe: str | None = None
    start: str | None = None
    evaluation_start: str | None = None
    evaluation_end: str | None = None
    rebalance_frequency: str | None = None
    strategy: str | None = None
    date: str | None = None
    cleaning_method: str | None = None
    correlation_input: str | None = None
    matrix_kind: str | None = None
    estimator_mode: str | None = None
    matrix_smoothing_span: int | None = None
    linear_shrinkage: float | None = None
    covariance_window: int | None = None
    weight_smoothing_alpha: float | None = None
    long_only: bool | None = None
    refresh_policy: str = 'auto'
    output_dir: str | None = None


@dataclass(frozen=True)
class InspectionSnapshotResult:
    request: InspectionSnapshotRequest
    universe: str
    cleaning_method: str
    correlation_input: str
    matrix_kind: str
    estimator_mode: str
    covariance_window: int
    allocation_date: pd.Timestamp
    sample_size: int
    num_assets: int
    sample_correlation: pd.DataFrame
    sample_covariance: pd.DataFrame
    empirical_cleaned_correlation: pd.DataFrame
    empirical_cleaned_covariance: pd.DataFrame
    cleaned_correlation: pd.DataFrame
    cleaned_covariance: pd.DataFrame
    correlation_spectrum: pd.DataFrame
    covariance_spectrum: pd.DataFrame
    correlation_eigenvectors: pd.DataFrame
    covariance_eigenvectors: pd.DataFrame
    feature_frame: pd.DataFrame
    cleaner_comparison_frame: pd.DataFrame
    artifacts: RunArtifacts
