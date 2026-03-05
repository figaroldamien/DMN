from __future__ import annotations

import math
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

    def build_dataset(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    for k in range(len(retrain_points) - 1):
        t0 = retrain_points[k]
        t1 = retrain_points[k + 1]
        train_mask = dates[dates < t0]
        test_mask = dates[(dates >= t0) & (dates < t1)]

        if len(train_mask) < cfg.min_obs:
            continue

        x_train, r_train, v_train = build_dataset(train_mask)
        if len(r_train) < 2000:
            continue

        # Refit a fresh model at each retraining date (pure walk-forward).
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

        net.eval()
        with torch.no_grad():
            idxs = np.where(test_mask)[0]
            for t_idx in idxs:
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
