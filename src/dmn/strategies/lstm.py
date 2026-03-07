"""LSTM strategy wrapper."""

from __future__ import annotations

import pandas as pd

from ..config import BacktestConfig
from .engine import run_walkforward_positions
from .models import LSTMPositionNet


def dmn_lstm_positions(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    retrain_years: int = 5,
    seq_len: int = 63,
    hidden: int = 32,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 256,
    turnover_lambda: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Run walk-forward training/inference with the baseline LSTM model."""

    return run_walkforward_positions(
        prices=prices,
        cfg=cfg,
        model_factory=LSTMPositionNet,
        model_kwargs={"n_features": None, "hidden": hidden, "dropout": dropout},
        retrain_years=retrain_years,
        seq_len=seq_len,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        turnover_lambda=turnover_lambda,
        seed=seed,
    )
