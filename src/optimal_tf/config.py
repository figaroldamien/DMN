from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseConfig:
    name: str = "cac40"
    start: str = "2000-01-01"


@dataclass(frozen=True)
class EstimationConfig:
    vol_span: int = 60
    covariance_alpha: float | None = None
    covariance_min_periods: int = 252
    corr_span: int = 252
    corr_min_periods: int = 252
    max_abs_return: float = 1.0
    cleaning_method: str = "empirical"
    linear_shrinkage: float = 0.0
    rie_bandwidth: float = 1e-3
    trend_alpha: float | None = None
    trend_span: int | None = 252
    torp_signal_gain: float = 1.0


@dataclass(frozen=True)
class BacktestConfig:
    sigma_target_annual: float = 0.15
    portfolio_vol_target: bool = True
    portfolio_vol_span: int = 60
    cost_bps: float = 0.0
    long_only: bool = False


@dataclass(frozen=True)
class AllocationConfig:
    strategy: str = "RP"
    date: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    strategy: str = "RP"
    rebalance_frequency: str = "monthly"
    evaluation_start: str | None = None
    evaluation_end: str | None = None
