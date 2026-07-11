from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import AllocationConfig, BacktestConfig, CompareConfig, EstimationConfig, EvaluationConfig, OutputConfig, UniverseConfig
from .features import alpha_from_span, effective_span_from_alpha
from .validation import validate_backtest_config, validate_estimation_config, validate_universe_config


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".toml":
        raise ValueError("Unsupported config format. Use .toml")
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_config(
    path: str | Path,
) -> tuple[UniverseConfig, EstimationConfig, BacktestConfig, AllocationConfig, EvaluationConfig, CompareConfig, OutputConfig]:
    raw = _read_mapping(Path(path))
    universe = UniverseConfig()
    estimation = EstimationConfig()
    backtest = BacktestConfig()
    allocation = AllocationConfig()
    evaluation = EvaluationConfig()
    compare = CompareConfig()
    output = OutputConfig()

    universe_raw = raw.get("universe", {}) if isinstance(raw.get("universe"), dict) else {}
    estimation_raw = raw.get("estimation", {}) if isinstance(raw.get("estimation"), dict) else {}
    backtest_raw = raw.get("backtest", {}) if isinstance(raw.get("backtest"), dict) else {}
    portfolio_raw = raw.get("portfolio", {}) if isinstance(raw.get("portfolio"), dict) else {}
    allocation_raw = raw.get("allocation", {}) if isinstance(raw.get("allocation"), dict) else {}
    evaluation_raw = raw.get("evaluation", {}) if isinstance(raw.get("evaluation"), dict) else {}
    compare_raw = raw.get("compare", {}) if isinstance(raw.get("compare"), dict) else {}
    output_raw = raw.get("output", {}) if isinstance(raw.get("output"), dict) else {}

    if backtest_raw and portfolio_raw:
        raise ValueError("Use either [backtest] or [portfolio], not both in the same config.")
    if portfolio_raw:
        backtest_raw = portfolio_raw

    if universe_raw:
        universe = replace(
            universe,
            **{
                k: universe_raw[k]
                for k in (
                    "name",
                    "start",
                    "quality_filter_enabled",
                    "quality_min_history_days",
                    "quality_min_coverage_ratio",
                    "quality_max_internal_missing",
                    "quality_max_abs_return",
                    "quality_require_latest_price",
                )
                if k in universe_raw
            },
        )
    if estimation_raw:
        estimation = replace(
            estimation,
            **{
                k: estimation_raw[k]
                for k in (
                    "vol_span",
                    "covariance_window",
                    "covariance_alpha",
                    "covariance_min_periods",
                    "corr_span",
                    "corr_min_periods",
                    "max_abs_return",
                    "cleaning_method",
                    "linear_shrinkage",
                    "rie_bandwidth",
                    "trend_alpha",
                    "trend_span",
                    "lltf_l2_reg",
                )
                if k in estimation_raw
            },
        )
        has_trend_alpha = "trend_alpha" in estimation_raw
        has_trend_span = "trend_span" in estimation_raw
        if has_trend_alpha and not has_trend_span:
            estimation = replace(estimation, trend_span=effective_span_from_alpha(estimation.trend_alpha))
        elif has_trend_span and not has_trend_alpha:
            estimation = replace(estimation, trend_alpha=alpha_from_span(estimation.trend_span))
    legacy_weight_smoothing_alpha = evaluation_raw.get("weight_smoothing_alpha") if "weight_smoothing_alpha" in evaluation_raw else None
    if backtest_raw:
        backtest = replace(
            backtest,
            **{
                k: backtest_raw[k]
                for k in (
                    "sigma_target_annual",
                    "portfolio_vol_target",
                    "portfolio_vol_span",
                    "cost_bps",
                    "weight_smoothing_alpha",
                    "long_only",
                )
                if k in backtest_raw
            },
        )
    if "weight_smoothing_alpha" not in backtest_raw and legacy_weight_smoothing_alpha is not None:
        backtest = replace(backtest, weight_smoothing_alpha=legacy_weight_smoothing_alpha)
    if allocation_raw:
        allocation = replace(allocation, **{k: allocation_raw[k] for k in ("strategy", "date") if k in allocation_raw})
    if evaluation_raw:
        evaluation = replace(
            evaluation,
            **{
                k: evaluation_raw[k]
                for k in ("strategy", "rebalance_frequency", "evaluation_start", "evaluation_end", "weight_smoothing_alpha")
                if k in evaluation_raw
            },
        )
    if compare_raw:
        strategies = compare_raw.get("strategies")
        if strategies is not None:
            if not isinstance(strategies, list) or not all(isinstance(item, str) for item in strategies):
                raise ValueError("[compare].strategies must be an array of strings")
            compare = replace(compare, strategies=tuple(strategies))
    if output_raw:
        output = replace(
            output,
            **{
                k: output_raw[k]
                for k in (
                    "allocation_csv",
                    "allocation_json",
                    "evaluation_dir",
                    "evaluation_plot",
                    "compare_dir",
                    "compare_clean_dir",
                    "compare_plot",
                )
                if k in output_raw
            },
        )

    validate_universe_config(universe)
    validate_estimation_config(estimation)
    validate_backtest_config(backtest)
    return universe, estimation, backtest, allocation, evaluation, compare, output
