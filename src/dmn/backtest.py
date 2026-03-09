from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import pandas as pd
import torch

from .portfolio import run_portfolio
from .config import BacktestConfig, ModelConfig
from .metrics import performance_metrics
from .strategies import (
    dmn_lstm_positions,
    ml_supervised_positions,
    strategy_baz_macd,
    strategy_long_only,
    strategy_sgn_12m,
    vlstm_positions,
    xlstm_positions,
)


def _evaluate_strategy_record(
    name: str,
    strategy_fn: Callable[..., pd.DataFrame],
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    *args: Any,
    **kwargs: Any,
) -> dict[str, float | str]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    x = strategy_fn(*args, **kwargs)
    strat, turnover, _ = run_portfolio(prices, x, cfg)
    perf = performance_metrics(strat, turnover)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    record: dict[str, float | str] = {
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
    print(f"{name}: Sharpe={perf.sharpe:.3f}, elapsed={elapsed:.2f}s")
    return record


def backtest_strategy(
    name: str,
    strategy_fn: Callable[..., pd.DataFrame],
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Evaluate one strategy independently and return a one-row metrics DataFrame.
    """
    record = _evaluate_strategy_record(name, strategy_fn, prices, cfg, *args, **kwargs)
    return pd.DataFrame([record])


def backtest_all(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    run_ml: bool = True,
    run_dmn: bool = True,
    model: ModelConfig | None = None,
) -> pd.DataFrame:
    model = model or ModelConfig()
    results = []
    results.append(_evaluate_strategy_record("LongOnly", strategy_long_only, prices, cfg, prices))
    results.append(_evaluate_strategy_record("Sgn12M", strategy_sgn_12m, prices, cfg, prices))
    results.append(_evaluate_strategy_record("MACD_Baz", strategy_baz_macd, prices, cfg, prices))

    if run_ml:
        try:
            results.append(
                _evaluate_strategy_record("ML_LassoReg", ml_supervised_positions, prices, cfg, prices, cfg, model_type="lasso_reg")
            )
            results.append(
                _evaluate_strategy_record("ML_MLPReg", ml_supervised_positions, prices, cfg, prices, cfg, model_type="mlp_reg")
            )
            results.append(
                _evaluate_strategy_record("ML_LassoClf", ml_supervised_positions, prices, cfg, prices, cfg, model_type="lasso_clf")
            )
        except Exception as e:
            warnings.warn(f"Skipping ML baselines due to: {e}")

    if run_dmn:
        try:
            if cfg.cost_bps > 0:
                results.append(
                    _evaluate_strategy_record(
                        "DMN_LSTM_Sharpe_TurnPen",
                        dmn_lstm_positions,
                        prices,
                        cfg,
                        prices,
                        cfg,
                        turnover_lambda=1e-2,
                        hidden=model.hidden,
                        dropout=model.dropout,
                        use_ticker_embedding=model.use_ticker_embedding,
                    )
                )
                results.append(
                    _evaluate_strategy_record(
                        "VLSTM_Sharpe_TurnPen",
                        vlstm_positions,
                        prices,
                        cfg,
                        prices,
                        cfg,
                        turnover_lambda=1e-2,
                        hidden=model.hidden,
                        dropout=model.dropout,
                        use_ticker_embedding=model.use_ticker_embedding,
                    )
                )
                results.append(
                    _evaluate_strategy_record(
                        "xLSTM_Sharpe_TurnPen",
                        xlstm_positions,
                        prices,
                        cfg,
                        prices,
                        cfg,
                        turnover_lambda=1e-2,
                        hidden=model.hidden,
                        dropout=model.dropout,
                        use_ticker_embedding=model.use_ticker_embedding,
                    )
                )
            else:
                results.append(
                    _evaluate_strategy_record(
                        "DMN_LSTM_Sharpe",
                        dmn_lstm_positions,
                        prices,
                        cfg,
                        prices,
                        cfg,
                        turnover_lambda=0.0,
                        hidden=model.hidden,
                        dropout=model.dropout,
                        use_ticker_embedding=model.use_ticker_embedding,
                    )
                )
                results.append(
                    _evaluate_strategy_record(
                        "VLSTM_Sharpe",
                        vlstm_positions,
                        prices,
                        cfg,
                        prices,
                        cfg,
                        turnover_lambda=0.0,
                        hidden=model.hidden,
                        dropout=model.dropout,
                        use_ticker_embedding=model.use_ticker_embedding,
                    )
                )
                results.append(
                    _evaluate_strategy_record(
                        "xLSTM_Sharpe",
                        xlstm_positions,
                        prices,
                        cfg,
                        prices,
                        cfg,
                        turnover_lambda=0.0,
                        hidden=model.hidden,
                        dropout=model.dropout,
                        use_ticker_embedding=model.use_ticker_embedding,
                    )
                )
        except Exception as e:
            warnings.warn(f"Skipping DMN due to: {e}")

    return pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
