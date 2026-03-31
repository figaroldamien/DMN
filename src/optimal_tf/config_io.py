from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import AllocationConfig, BacktestConfig, EstimationConfig, EvaluationConfig, UniverseConfig


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".toml":
        raise ValueError("Unsupported config format. Use .toml")
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_config(
    path: str | Path,
) -> tuple[UniverseConfig, EstimationConfig, BacktestConfig, AllocationConfig, EvaluationConfig]:
    raw = _read_mapping(Path(path))
    universe = UniverseConfig()
    estimation = EstimationConfig()
    backtest = BacktestConfig()
    allocation = AllocationConfig()
    evaluation = EvaluationConfig()

    universe_raw = raw.get("universe", {}) if isinstance(raw.get("universe"), dict) else {}
    estimation_raw = raw.get("estimation", {}) if isinstance(raw.get("estimation"), dict) else {}
    backtest_raw = raw.get("backtest", {}) if isinstance(raw.get("backtest"), dict) else {}
    allocation_raw = raw.get("allocation", {}) if isinstance(raw.get("allocation"), dict) else {}
    evaluation_raw = raw.get("evaluation", {}) if isinstance(raw.get("evaluation"), dict) else {}

    if universe_raw:
        universe = replace(universe, **{k: universe_raw[k] for k in ("name", "start") if k in universe_raw})
    if estimation_raw:
        estimation = replace(
            estimation,
            **{
                k: estimation_raw[k]
                for k in (
                    "vol_span",
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
                    "torp_signal_gain",
                )
                if k in estimation_raw
            },
        )
    if backtest_raw:
        backtest = replace(
            backtest,
            **{
                k: backtest_raw[k]
                for k in ("sigma_target_annual", "portfolio_vol_target", "portfolio_vol_span", "cost_bps", "long_only")
                if k in backtest_raw
            },
        )
    if allocation_raw:
        allocation = replace(allocation, **{k: allocation_raw[k] for k in ("strategy", "date") if k in allocation_raw})
    if evaluation_raw:
        evaluation = replace(
            evaluation,
            **{
                k: evaluation_raw[k]
                for k in ("strategy", "rebalance_frequency", "evaluation_start", "evaluation_end")
                if k in evaluation_raw
            },
        )

    return universe, estimation, backtest, allocation, evaluation
