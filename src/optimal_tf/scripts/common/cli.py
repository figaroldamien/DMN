from __future__ import annotations

from dataclasses import replace

import pandas as pd

from optimal_tf.allocation import supported_strategies
from optimal_tf.config import BacktestConfig, EstimationConfig, EvaluationConfig, UniverseConfig
from trading_core.rebalance import resolve_rebalance_dates
from trading_core.risk import supported_cleaning_methods


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_windows(raw: str) -> list[int]:
    windows = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not windows:
        raise ValueError("At least one covariance window must be provided.")
    if any(window <= 1 for window in windows):
        raise ValueError(f"All covariance windows must be > 1 (got {windows}).")
    return windows


def merge_common_overrides(
    universe: UniverseConfig,
    estimation: EstimationConfig,
    backtest: BacktestConfig,
    evaluation: EvaluationConfig,
    args: object,
) -> tuple[UniverseConfig, EstimationConfig, BacktestConfig, EvaluationConfig]:
    if getattr(args, "universe", None) is not None:
        universe = UniverseConfig(name=args.universe, start=universe.start)
    if getattr(args, "start", None) is not None:
        universe = UniverseConfig(name=universe.name, start=args.start)
    covariance_window = getattr(args, "covariance_window", None)
    covariance_min_periods = getattr(args, "covariance_min_periods", None)
    if covariance_window is not None:
        estimation = replace(estimation, covariance_window=int(covariance_window), corr_span=int(covariance_window))
    if covariance_min_periods is not None:
        estimation = replace(
            estimation,
            covariance_min_periods=int(covariance_min_periods),
            corr_min_periods=int(covariance_min_periods),
        )
    evaluation = replace(
        evaluation,
        rebalance_frequency=getattr(args, "rebalance_frequency", None) or evaluation.rebalance_frequency,
        evaluation_start=getattr(args, "evaluation_start", None) or evaluation.evaluation_start,
        evaluation_end=getattr(args, "evaluation_end", None) or evaluation.evaluation_end,
    )
    return universe, estimation, backtest, evaluation


def resolve_target_dates(prices: pd.DataFrame, evaluation: EvaluationConfig) -> pd.DatetimeIndex:
    return resolve_rebalance_dates(
        prices.index,
        evaluation.rebalance_frequency,
        start=evaluation.evaluation_start,
        end=evaluation.evaluation_end,
    )


def validate_methods(methods: list[str]) -> None:
    allowed = list(supported_cleaning_methods())
    invalid = [method for method in methods if method not in allowed]
    if invalid:
        raise ValueError(f"Unknown cleaning methods {invalid}. Allowed values: {allowed}")


def validate_strategies(strategies: list[str]) -> None:
    allowed = supported_strategies()
    invalid = [strategy for strategy in strategies if strategy not in allowed]
    if invalid:
        raise ValueError(f"Unknown strategies {invalid}. Allowed values: {allowed}")


def _resolve_min_periods(window: int, base_min_periods: int, mode: str) -> int:
    if mode == "fixed":
        return base_min_periods
    return min(base_min_periods, window)


def resolve_window_estimation_cfg(
    estimation: EstimationConfig,
    window: int,
    *,
    min_periods_mode: str,
) -> EstimationConfig:
    return replace(
        estimation,
        covariance_window=window,
        corr_span=window,
        covariance_min_periods=_resolve_min_periods(window, estimation.covariance_min_periods, min_periods_mode),
        corr_min_periods=_resolve_min_periods(window, estimation.corr_min_periods, min_periods_mode),
    )
