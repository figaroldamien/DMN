from __future__ import annotations

import itertools
import time
from dataclasses import asdict
from typing import Callable

import pandas as pd
import torch

from .config import BacktestConfig, OptimizationConfig
from .metrics import performance_metrics
from .portfolio import run_portfolio
from .strategies import dmn_lstm_positions, vlstm_positions, xlstm_positions


def strategy_registry() -> dict[str, tuple[Callable[..., pd.DataFrame], dict]]:
    return {
        "DMN_LSTM_Sharpe": (dmn_lstm_positions, {"turnover_lambda": 0.0}),
        "DMN_LSTM_Sharpe_TurnPen": (dmn_lstm_positions, {"turnover_lambda": 1e-2}),
        "VLSTM_Sharpe": (vlstm_positions, {"turnover_lambda": 0.0}),
        "VLSTM_Sharpe_TurnPen": (vlstm_positions, {"turnover_lambda": 1e-2}),
        "xLSTM_Sharpe": (xlstm_positions, {"turnover_lambda": 0.0}),
        "xLSTM_Sharpe_TurnPen": (xlstm_positions, {"turnover_lambda": 1e-2}),
    }


def validate_optimization_config(cfg: OptimizationConfig) -> None:
    if cfg.strategy not in strategy_registry():
        raise ValueError(
            f"Unknown optimization strategy '{cfg.strategy}'. Allowed values: {sorted(strategy_registry().keys())}"
        )
    if not cfg.metric:
        raise ValueError("Optimization metric is required.")
    required_lists = {
        "hidden_values": cfg.hidden_values,
        "dropout_values": cfg.dropout_values,
        "batch_size_values": cfg.batch_size_values,
        "learning_rate_values": cfg.learning_rate_values,
        "epochs_values": cfg.epochs_values,
    }
    for key, values in required_lists.items():
        if not values:
            raise ValueError(f"Optimization grid '{key}' is required and must be non-empty.")


def iter_grid(cfg: OptimizationConfig):
    for hidden, dropout, batch_size, learning_rate, epochs in itertools.product(
        cfg.hidden_values,
        cfg.dropout_values,
        cfg.batch_size_values,
        cfg.learning_rate_values,
        cfg.epochs_values,
    ):
        yield {
            "hidden": hidden,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
        }


def _format_candidate_progress(
    optimization_cfg: OptimizationConfig,
    hyperparams: dict[str, int | float],
    candidate_idx: int | None = None,
    total_candidates: int | None = None,
) -> str:
    prefix = ""
    if candidate_idx is not None and total_candidates is not None:
        prefix = f"[{candidate_idx}/{total_candidates}] "
    return (
        f"{prefix}"
        f"h={int(hyperparams['hidden'])} "
        f"d={float(hyperparams['dropout']):.3f} "
        f"bs={int(hyperparams['batch_size'])} "
        f"lr={float(hyperparams['learning_rate']):.6g} "
        f"ep={int(hyperparams['epochs'])}"
    )


def _format_candidate_result(
    optimization_cfg: OptimizationConfig,
    record: dict[str, int | float | str],
) -> str:
    metric = optimization_cfg.metric
    metric_value = record.get(metric)
    metric_display = f"{metric}=n/a"
    if isinstance(metric_value, (int, float)):
        metric_display = f"{metric}={metric_value:.3f}"

    elapsed_value = record.get("elapsed_s")
    elapsed_display = "t=n/a"
    if isinstance(elapsed_value, (int, float)):
        elapsed_display = f"t={elapsed_value:.1f}s"

    return f"{metric_display} {elapsed_display}"


def evaluate_candidate(
    prices: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    optimization_cfg: OptimizationConfig,
    hyperparams: dict[str, int | float],
    candidate_idx: int | None = None,
    total_candidates: int | None = None,
) -> dict[str, int | float | str]:
    strategy_fn, strategy_kwargs = strategy_registry()[optimization_cfg.strategy]
    progress_prefix = _format_candidate_progress(
        optimization_cfg,
        hyperparams,
        candidate_idx,
        total_candidates,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    positions = strategy_fn(
        prices,
        backtest_cfg,
        hidden=int(hyperparams["hidden"]),
        dropout=float(hyperparams["dropout"]),
        lr=float(hyperparams["learning_rate"]),
        epochs=int(hyperparams["epochs"]),
        batch_size=int(hyperparams["batch_size"]),
        **strategy_kwargs,
    )
    strat, turnover, _ = run_portfolio(prices, positions, backtest_cfg)
    perf = performance_metrics(strat, turnover)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    record: dict[str, int | float | str] = {
        "strategy": optimization_cfg.strategy,
        **hyperparams,
        "ann_return": perf.ann_return,
        "ann_vol": perf.ann_vol,
        "sharpe": perf.sharpe,
        "sortino": perf.sortino,
        "calmar": perf.calmar,
        "mdd": perf.mdd,
        "pct_pos": perf.pct_pos,
        "avgP_over_avgL": perf.avg_profit_over_avg_loss,
        "avg_turnover": perf.avg_turnover,
        "elapsed_s": elapsed,
    }
    print(f"{progress_prefix} -> {_format_candidate_result(optimization_cfg, record)}", flush=True)
    return record


def run_grid_search(
    prices: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    optimization_cfg: OptimizationConfig,
) -> pd.DataFrame:
    validate_optimization_config(optimization_cfg)

    grid = list(iter_grid(optimization_cfg))
    total_candidates = len(grid)
    records = [
        evaluate_candidate(
            prices,
            backtest_cfg,
            optimization_cfg,
            hyperparams,
            candidate_idx=idx,
            total_candidates=total_candidates,
        )
        for idx, hyperparams in enumerate(grid, start=1)
    ]
    if not records:
        return pd.DataFrame()

    metric = optimization_cfg.metric
    results = pd.DataFrame(records)
    if metric not in results.columns:
        raise ValueError(
            f"Unknown optimization metric '{metric}'. Available metrics: {sorted(results.columns)}"
        )
    return results.sort_values(metric, ascending=False).reset_index(drop=True)


def optimization_summary(results: pd.DataFrame) -> dict[str, int | float | str]:
    if results.empty:
        raise ValueError("Optimization results are empty.")
    return results.iloc[0].to_dict()


def optimization_config_to_dict(cfg: OptimizationConfig) -> dict[str, object]:
    return asdict(cfg)
