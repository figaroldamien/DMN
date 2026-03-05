from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import pandas as pd
import torch

from .backtest import run_portfolio
from .config import BacktestConfig
from .metrics import performance_metrics
from .strategies import (
    dmn_lstm_positions,
    ml_supervised_positions,
    strategy_baz_macd,
    strategy_long_only,
    strategy_sgn_12m,
)


def backtest_all(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    run_ml: bool = True,
    run_dmn: bool = True,
) -> pd.DataFrame:
    results = []

    def eval_strategy(name: str, strategy_fn: Callable[..., pd.DataFrame], *args: Any, **kwargs: Any):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        x = strategy_fn(*args, **kwargs)
        strat, turnover, _ = run_portfolio(prices, x, cfg)
        perf = performance_metrics(strat, turnover)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        results.append(
            {
                "strategy": name,
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
        )
        print(f"{name}: Sharpe={perf.sharpe:.3f}, elapsed={elapsed:.2f}s")

    eval_strategy("LongOnly", strategy_long_only, prices)
    eval_strategy("Sgn12M", strategy_sgn_12m, prices)
    eval_strategy("MACD_Baz", strategy_baz_macd, prices)

    if run_ml:
        try:
            eval_strategy("ML_LassoReg", ml_supervised_positions, prices, cfg, model_type="lasso_reg")
            eval_strategy("ML_MLPReg", ml_supervised_positions, prices, cfg, model_type="mlp_reg")
            eval_strategy("ML_LassoClf", ml_supervised_positions, prices, cfg, model_type="lasso_clf")
        except Exception as e:
            warnings.warn(f"Skipping ML baselines due to: {e}")

    if run_dmn:
        try:
            if cfg.cost_bps > 0:
                eval_strategy("DMN_LSTM_Sharpe_TurnPen", dmn_lstm_positions, prices, cfg, turnover_lambda=1e-2)
            else:
                eval_strategy("DMN_LSTM_Sharpe", dmn_lstm_positions, prices, cfg, turnover_lambda=0.0)
        except Exception as e:
            warnings.warn(f"Skipping DMN due to: {e}")

    return pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
