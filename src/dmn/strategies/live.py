"""Live/offline lifecycle for the baseline DMN LSTM model.

This module handles:
- training up to a cutoff date,
- saving/loading model artifacts,
- inference for future/live dates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import BacktestConfig
from .artifacts import DMNLSTMArtifact
from .engine import build_dataset, prepare_sequence_data, set_seed
from .models import LSTMPositionNet
from .engine import fit_model_sharpe


def train_lstm_until_cutoff(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    cutoff_date: str | pd.Timestamp,
    seq_len: int = 63,
    hidden: int = 32,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 256,
    turnover_lambda: float = 0.0,
    seed: int = 0,
    min_train_samples: int = 2000,
) -> tuple[LSTMPositionNet, DMNLSTMArtifact]:
    """Train a baseline LSTM strategy up to `cutoff_date` (inclusive)."""

    set_seed(seed)

    cutoff_ts = pd.Timestamp(cutoff_date)
    train_prices = prices.loc[:cutoff_ts].copy()
    if train_prices.empty:
        raise ValueError("No prices available on or before cutoff_date.")
    if len(train_prices.index) < cfg.min_obs:
        raise ValueError(
            f"Not enough history before cutoff_date ({len(train_prices.index)} < min_obs={cfg.min_obs})."
        )

    data = prepare_sequence_data(train_prices, cfg, seq_len=seq_len)
    train_mask = data.dates <= cutoff_ts
    x_train, r_train, v_train = build_dataset(
        train_mask,
        data.dates,
        data.syms,
        data.get_feat_matrix,
        data.next_ret,
        data.daily_vol,
        data.seq_len,
        data.n_features,
    )
    if len(r_train) < min_train_samples:
        raise ValueError(
            f"Insufficient train samples ({len(r_train)} < min_train_samples={min_train_samples})."
        )

    model = fit_model_sharpe(
        model_factory=LSTMPositionNet,
        model_kwargs={"n_features": data.n_features, "hidden": hidden, "dropout": dropout},
        x_train=x_train,
        r_train=r_train,
        v_train=v_train,
        sigma_tgt_daily=data.sigma_tgt_daily,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        turnover_lambda=turnover_lambda,
    )

    artifact = DMNLSTMArtifact(
        tickers=data.syms,
        feature_names=data.feat_names,
        seq_len=seq_len,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        turnover_lambda=turnover_lambda,
        seed=seed,
        vol_span=cfg.vol_span,
        sigma_target_annual=cfg.sigma_target_annual,
        cost_bps=cfg.cost_bps,
        portfolio_vol_target=cfg.portfolio_vol_target,
        min_obs=cfg.min_obs,
        cutoff_date=str(pd.Timestamp(cutoff_ts).date()),
        train_start_date=str(pd.Timestamp(data.dates[0]).date()),
        train_end_date=str(pd.Timestamp(data.dates[-1]).date()),
        n_train_samples=int(len(r_train)),
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return model, artifact


def save_lstm_artifact(
    model: LSTMPositionNet,
    artifact: DMNLSTMArtifact,
    artifact_dir: str | Path,
    artifact_name: str | None = None,
) -> Path:
    """Persist model state and metadata to a `.pt` artifact file."""

    base_dir = Path(artifact_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    name = artifact_name or f"dmn_lstm_{artifact.cutoff_date.replace('-', '')}"
    out_path = base_dir / f"{name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": artifact.to_dict(),
        },
        out_path,
    )
    return out_path


def load_lstm_artifact(path: str | Path) -> tuple[LSTMPositionNet, DMNLSTMArtifact]:
    """Load a previously saved DMN LSTM artifact from disk."""

    payload = torch.load(Path(path), map_location="cpu")
    artifact = DMNLSTMArtifact.from_dict(payload["metadata"])
    model = LSTMPositionNet(
        n_features=len(artifact.feature_names),
        hidden=artifact.hidden,
        dropout=artifact.dropout,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, artifact


def predict_positions_from_model(
    prices: pd.DataFrame,
    model: LSTMPositionNet,
    artifact: DMNLSTMArtifact,
    from_date: str | pd.Timestamp | None = None,
    to_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate positions from a loaded artifact for a target date range."""

    missing = [t for t in artifact.tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers required by artifact: {missing}")

    prices = prices.loc[:, artifact.tickers].copy()
    cfg_proxy = BacktestConfig(vol_span=artifact.vol_span, sigma_target_annual=artifact.sigma_target_annual)
    data = prepare_sequence_data(prices, cfg_proxy, seq_len=artifact.seq_len)

    dates = data.dates
    start_ts = pd.Timestamp(from_date) if from_date is not None else pd.Timestamp(dates[-1])
    end_ts = pd.Timestamp(to_date) if to_date is not None else pd.Timestamp(dates[-1])
    infer_mask = (dates >= start_ts) & (dates <= end_ts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    positions = pd.DataFrame(index=prices.index, columns=artifact.tickers, dtype=float)
    with torch.no_grad():
        for t_idx in np.where(infer_mask)[0]:
            t = dates[t_idx]
            for sym in artifact.tickers:
                x = data.get_feat_matrix(t_idx, sym)
                if x is None:
                    continue
                xb = torch.tensor(x[None, :, :], device=device)
                pos = float(model(xb).cpu().numpy()[0])
                positions.loc[t, sym] = pos

    positions = positions.loc[infer_mask].ffill().fillna(0.0).clip(-1.0, 1.0)
    return positions
