from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import BacktestConfig
from ..features import compute_returns, ewma_vol, make_dmn_features


class LSTMPositionNet(nn.Module):
    # Sequence model that outputs a continuous position in [-1, 1].
    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        pos = torch.tanh(self.fc(out))
        return pos.squeeze(-1)


def sharpe_loss(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Directly optimize risk-adjusted return (negative Sharpe for minimization).
    mu = returns.mean()
    sd = returns.std(unbiased=False)
    return -(mu / (sd + eps)) * math.sqrt(252)


@dataclass
class DMNLSTMArtifact:
    tickers: list[str]
    feature_names: list[str]
    seq_len: int
    hidden: int
    dropout: float
    lr: float
    epochs: int
    batch_size: int
    turnover_lambda: float
    seed: int
    vol_span: int
    sigma_target_annual: float
    cutoff_date: str
    train_start_date: str
    train_end_date: str
    n_train_samples: int
    trained_at_utc: str
    cost_bps: float = 0.0
    portfolio_vol_target: bool = True
    min_obs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict) -> "DMNLSTMArtifact":
        defaults = {
            "cost_bps": 0.0,
            "portfolio_vol_target": True,
            "min_obs": 0,
        }
        for k, v in defaults.items():
            raw.setdefault(k, v)
        return DMNLSTMArtifact(**raw)


def _make_feature_extractor(
    feats: pd.DataFrame,
    feat_names: list[str],
    seq_len: int,
    n_features: int,
):
    def get_feat_matrix(t_idx: int, sym: str) -> Optional[np.ndarray]:
        # Build one supervised sample: sequence of shape (seq_len, n_features).
        if t_idx - seq_len + 1 < 0:
            return None
        window = feats.iloc[t_idx - seq_len + 1 : t_idx + 1]
        arr = np.zeros((seq_len, n_features), dtype=np.float32)
        for j, fn in enumerate(feat_names):
            arr[:, j] = window[(fn, sym)].values.astype(np.float32)
        if np.any(np.isnan(arr)):
            return None
        return arr

    return get_feat_matrix


def _build_dataset(
    mask: np.ndarray,
    dates: pd.Index,
    syms: list[str],
    get_feat_matrix,
    next_ret: pd.DataFrame,
    daily_vol: pd.DataFrame,
    seq_len: int,
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Flatten panel data into independent training samples over (date, asset).
    x_list, r_list, vol_list = [], [], []
    idxs = np.where(mask)[0]
    for t_idx in idxs:
        t = dates[t_idx]
        for sym in syms:
            x = get_feat_matrix(t_idx, sym)
            if x is None:
                continue
            r = next_ret.loc[t, sym]
            v = daily_vol.loc[t, sym]
            if np.isnan(r) or np.isnan(v) or v <= 0:
                continue
            x_list.append(x)
            r_list.append(r)
            vol_list.append(v)
    if not x_list:
        return (
            np.empty((0, seq_len, n_features), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.float32),
        )
    return np.stack(x_list), np.asarray(r_list, np.float32), np.asarray(vol_list, np.float32)


def _fit_single_model(
    x_train: np.ndarray,
    r_train: np.ndarray,
    v_train: np.ndarray,
    n_features: int,
    sigma_tgt_daily: float,
    hidden: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    turnover_lambda: float,
) -> LSTMPositionNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LSTMPositionNet(n_features=n_features, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    n = len(r_train)
    order = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(order)
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            xb = torch.tensor(x_train[idx], device=device)
            rb = torch.tensor(r_train[idx], device=device)
            vb = torch.tensor(v_train[idx], device=device)

            pos = net(xb)
            # Approximate captured return under per-asset vol scaling.
            ret_cap = pos * (sigma_tgt_daily / (vb + 1e-12)) * rb

            loss = sharpe_loss(ret_cap)
            if turnover_lambda > 0:
                # Proxy regularizer to discourage excessive position magnitude.
                loss = loss + turnover_lambda * pos.abs().mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

    return net


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
    """
    Train a single DMN LSTM on historical data up to `cutoff_date` (inclusive).
    Returns the trained model and an artifact descriptor for persistence.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cutoff_ts = pd.Timestamp(cutoff_date)
    train_prices = prices.loc[:cutoff_ts].copy()
    if train_prices.empty:
        raise ValueError("No prices available on or before cutoff_date.")
    if len(train_prices.index) < cfg.min_obs:
        raise ValueError(
            f"Not enough history before cutoff_date ({len(train_prices.index)} < min_obs={cfg.min_obs})."
        )

    rets = compute_returns(train_prices)
    daily_vol = ewma_vol(rets, span=cfg.vol_span)
    feats = make_dmn_features(train_prices, daily_vol)
    next_ret = rets.shift(-1)
    sigma_tgt_daily = cfg.sigma_target_annual / math.sqrt(252)

    dates = train_prices.index
    syms = list(train_prices.columns)
    feat_names = sorted({c[0] for c in feats.columns})
    n_features = len(feat_names)

    get_feat_matrix = _make_feature_extractor(feats, feat_names, seq_len, n_features)
    train_mask = dates <= cutoff_ts
    x_train, r_train, v_train = _build_dataset(
        train_mask,
        dates,
        syms,
        get_feat_matrix,
        next_ret,
        daily_vol,
        seq_len,
        n_features,
    )
    if len(r_train) < min_train_samples:
        raise ValueError(
            f"Insufficient train samples ({len(r_train)} < min_train_samples={min_train_samples})."
        )

    model = _fit_single_model(
        x_train=x_train,
        r_train=r_train,
        v_train=v_train,
        n_features=n_features,
        sigma_tgt_daily=sigma_tgt_daily,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        turnover_lambda=turnover_lambda,
    )

    artifact = DMNLSTMArtifact(
        tickers=syms,
        feature_names=feat_names,
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
        train_start_date=str(pd.Timestamp(dates[0]).date()),
        train_end_date=str(pd.Timestamp(dates[-1]).date()),
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
    """
    Generate DMN positions for future/live dates using a persisted model.
    The input `prices` must include enough lookback to compute features and sequences.
    """
    missing = [t for t in artifact.tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Missing tickers required by artifact: {missing}")

    prices = prices.loc[:, artifact.tickers].copy()
    cfg_proxy = BacktestConfig(vol_span=artifact.vol_span, sigma_target_annual=artifact.sigma_target_annual)

    rets = compute_returns(prices)
    daily_vol = ewma_vol(rets, span=cfg_proxy.vol_span)
    feats = make_dmn_features(prices, daily_vol)

    feat_names = artifact.feature_names
    n_features = len(feat_names)
    get_feat_matrix = _make_feature_extractor(feats, feat_names, artifact.seq_len, n_features)

    dates = prices.index
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
                x = get_feat_matrix(t_idx, sym)
                if x is None:
                    continue
                xb = torch.tensor(x[None, :, :], device=device)
                pos = float(model(xb).cpu().numpy()[0])
                positions.loc[t, sym] = pos

    positions = positions.loc[infer_mask].ffill().fillna(0.0).clip(-1.0, 1.0)
    return positions


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
    """
    Walk-forward DMN training and inference.

    For each retraining window:
    1) Build per-asset rolling feature sequences.
    2) Train LSTM with Sharpe-based objective on next-day returns.
    3) Infer positions on the next test window.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Shared feature/target tensors used across all walk-forward windows.
    rets = compute_returns(prices)
    daily_vol = ewma_vol(rets, span=cfg.vol_span)
    feats = make_dmn_features(prices, daily_vol)

    next_ret = rets.shift(-1)
    sigma_tgt_daily = cfg.sigma_target_annual / math.sqrt(252)

    dates = prices.index
    start_date = dates[0]
    end_date = dates[-1]

    retrain_points = []
    cur = start_date + pd.DateOffset(years=retrain_years)
    while cur < end_date:
        retrain_points.append(cur)
        cur = cur + pd.DateOffset(years=retrain_years)
    retrain_points.append(end_date + pd.Timedelta(days=1))

    positions = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    feat_names = sorted({c[0] for c in feats.columns})
    syms = list(prices.columns)
    n_features = len(feat_names)
    get_feat_matrix = _make_feature_extractor(feats, feat_names, seq_len, n_features)

    for k in range(len(retrain_points) - 1):
        t0 = retrain_points[k]
        t1 = retrain_points[k + 1]
        train_mask = dates < t0
        test_mask = (dates >= t0) & (dates < t1)

        if int(train_mask.sum()) < cfg.min_obs:
            continue

        x_train, r_train, v_train = _build_dataset(
            train_mask,
            dates,
            syms,
            get_feat_matrix,
            next_ret,
            daily_vol,
            seq_len,
            n_features,
        )
        if len(r_train) < 2000:
            continue

        # Refit a fresh model at each retraining date (pure walk-forward).
        net = _fit_single_model(
            x_train=x_train,
            r_train=r_train,
            v_train=v_train,
            n_features=n_features,
            sigma_tgt_daily=sigma_tgt_daily,
            hidden=hidden,
            dropout=dropout,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            turnover_lambda=turnover_lambda,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        net.eval()
        with torch.no_grad():
            for t_idx in np.where(test_mask)[0]:
                t = dates[t_idx]
                for sym in syms:
                    x = get_feat_matrix(t_idx, sym)
                    if x is None:
                        continue
                    # Inference emits one position per (date, asset).
                    xb = torch.tensor(x[None, :, :], device=device)
                    pos = float(net(xb).cpu().numpy()[0])
                    positions.loc[t, sym] = pos

        # Hold last known signal when a new point cannot be computed.
        positions = positions.ffill()

    # Keep output inside the expected trading signal bounds.
    return positions.fillna(0.0).clip(-1.0, 1.0)
