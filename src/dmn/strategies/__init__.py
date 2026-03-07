"""Public strategy exports used by backtest and CLI entrypoints."""

from .artifacts import DMNLSTMArtifact
from .baseline import strategy_baz_macd, strategy_long_only, strategy_sgn_12m
from .engine import sharpe_loss
from .lstm import dmn_lstm_positions
from .live import (
    load_lstm_artifact,
    predict_positions_from_model,
    save_lstm_artifact,
    train_lstm_until_cutoff,
)
from .ml import ml_supervised_positions
from .models import LSTMPositionNet, VLSTMPositionNet, xLSTMPositionNet
from .vlstm import vlstm_positions
from .xlstm import xlstm_positions

__all__ = [
    "DMNLSTMArtifact",
    "LSTMPositionNet",
    "dmn_lstm_positions",
    "load_lstm_artifact",
    "ml_supervised_positions",
    "predict_positions_from_model",
    "save_lstm_artifact",
    "sharpe_loss",
    "strategy_baz_macd",
    "strategy_long_only",
    "strategy_sgn_12m",
    "train_lstm_until_cutoff",
    "VLSTMPositionNet",
    "vlstm_positions",
    "xLSTMPositionNet",
    "xlstm_positions",
]
