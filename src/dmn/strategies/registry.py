from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .baseline import strategy_baz_macd, strategy_long_only, strategy_sgn_12m
from .lstm import dmn_lstm_positions
from .ml import ml_supervised_positions
from .vlstm import vlstm_positions
from .xlstm import xlstm_positions


@dataclass(frozen=True)
class StrategySpec:
    name: str
    fn: Callable[..., pd.DataFrame]
    default_kwargs: dict[str, Any]
    cli_argsets: tuple[str, ...] = ()
    supports_optimization: bool = False


def strategy_specs() -> dict[str, StrategySpec]:
    return {
        "LongOnly": StrategySpec(
            name="LongOnly",
            fn=strategy_long_only,
            default_kwargs={},
        ),
        "Sgn12M": StrategySpec(
            name="Sgn12M",
            fn=strategy_sgn_12m,
            default_kwargs={},
        ),
        "MACD_Baz": StrategySpec(
            name="MACD_Baz",
            fn=strategy_baz_macd,
            default_kwargs={},
        ),
        "ML_LassoReg": StrategySpec(
            name="ML_LassoReg",
            fn=ml_supervised_positions,
            default_kwargs={"model_type": "lasso_reg"},
        ),
        "ML_MLPReg": StrategySpec(
            name="ML_MLPReg",
            fn=ml_supervised_positions,
            default_kwargs={"model_type": "mlp_reg"},
        ),
        "ML_LassoClf": StrategySpec(
            name="ML_LassoClf",
            fn=ml_supervised_positions,
            default_kwargs={"model_type": "lasso_clf"},
        ),
        "DMN_LSTM_Sharpe": StrategySpec(
            name="DMN_LSTM_Sharpe",
            fn=dmn_lstm_positions,
            default_kwargs={"turnover_lambda": 0.0},
            cli_argsets=("model", "optimization"),
            supports_optimization=True,
        ),
        "DMN_LSTM_Sharpe_TurnPen": StrategySpec(
            name="DMN_LSTM_Sharpe_TurnPen",
            fn=dmn_lstm_positions,
            default_kwargs={"turnover_lambda": 1e-2},
            cli_argsets=("model", "optimization"),
            supports_optimization=True,
        ),
        "VLSTM_Sharpe": StrategySpec(
            name="VLSTM_Sharpe",
            fn=vlstm_positions,
            default_kwargs={"turnover_lambda": 0.0},
            cli_argsets=("model", "optimization"),
            supports_optimization=True,
        ),
        "VLSTM_Sharpe_TurnPen": StrategySpec(
            name="VLSTM_Sharpe_TurnPen",
            fn=vlstm_positions,
            default_kwargs={"turnover_lambda": 1e-2},
            cli_argsets=("model", "optimization"),
            supports_optimization=True,
        ),
        "xLSTM_Sharpe": StrategySpec(
            name="xLSTM_Sharpe",
            fn=xlstm_positions,
            default_kwargs={"turnover_lambda": 0.0},
            cli_argsets=("model", "optimization"),
            supports_optimization=True,
        ),
        "xLSTM_Sharpe_TurnPen": StrategySpec(
            name="xLSTM_Sharpe_TurnPen",
            fn=xlstm_positions,
            default_kwargs={"turnover_lambda": 1e-2},
            cli_argsets=("model", "optimization"),
            supports_optimization=True,
        ),
    }


def get_strategy_spec(name: str) -> StrategySpec:
    specs = strategy_specs()
    if name not in specs:
        raise KeyError(f"Unknown strategy '{name}'. Allowed values: {sorted(specs)}")
    return specs[name]


def strategy_names(*, supports_optimization: bool | None = None) -> list[str]:
    specs = strategy_specs().values()
    if supports_optimization is not None:
        specs = [spec for spec in specs if spec.supports_optimization == supports_optimization]
    return sorted(spec.name for spec in specs)


def strategy_registry(*, supports_optimization: bool | None = None) -> dict[str, tuple[Callable[..., pd.DataFrame], dict[str, Any]]]:
    specs = strategy_specs().values()
    if supports_optimization is not None:
        specs = [spec for spec in specs if spec.supports_optimization == supports_optimization]
    return {spec.name: (spec.fn, dict(spec.default_kwargs)) for spec in specs}
