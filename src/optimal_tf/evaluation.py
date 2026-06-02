from .allocation import compute_strategy_panel
from .config import BacktestConfig, EstimationConfig, EvaluationConfig
from trading_core.backtest import EvaluationResult
from trading_core.backtest.engine import (
    apply_portfolio_vol_target as _engine_apply_portfolio_vol_target,
    evaluate_portfolio as _engine_evaluate_portfolio,
    slice_next_holding_period as _slice_next_holding_period,
)
from trading_core.risk import estimate_clean_covariance_panel


def _apply_portfolio_vol_target(
    gross_returns,
    bt_cfg,
):
    return _engine_apply_portfolio_vol_target(gross_returns, bt_cfg)


def evaluate_portfolio(
    prices,
    est_cfg,
    bt_cfg,
    eval_cfg,
):
    return _engine_evaluate_portfolio(
        prices,
        est_cfg,
        bt_cfg,
        eval_cfg,
        compute_strategy_panel_fn=compute_strategy_panel,
        estimate_clean_covariance_panel_fn=estimate_clean_covariance_panel,
    )


__all__ = [
    "EvaluationResult",
    "_apply_portfolio_vol_target",
    "_slice_next_holding_period",
    "compute_strategy_panel",
    "estimate_clean_covariance_panel",
    "evaluate_portfolio",
]
