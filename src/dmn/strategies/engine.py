"""Shared sequence-strategy engine.

Contains the common data preparation, training loop, and walk-forward execution
used by `dmn_lstm`, `vlstm`, and `xlstm`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import BacktestConfig
from ..features import compute_returns, ewma_vol, make_dmn_features


@dataclass
class SequenceDataBundle:
    """Container for precomputed tensors and metadata used by sequence models."""

    dates: pd.Index
    syms: list[str]
    sym_to_idx: dict[str, int]
    feat_names: list[str]
    n_features: int
    seq_len: int
    next_ret: pd.DataFrame
    daily_vol: pd.DataFrame
    sigma_tgt_daily: float
    get_feat_matrix: Callable[[int, str], Optional[np.ndarray]]


def set_seed(seed: int) -> None:
    """Set NumPy and PyTorch RNG seeds for reproducibility."""

    torch.manual_seed(seed)
    np.random.seed(seed)


def sharpe_loss(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative annualized Sharpe ratio (minimization objective)."""

    mu = returns.mean()
    sd = returns.std(unbiased=False)
    return -(mu / (sd + eps)) * math.sqrt(252)


def make_feature_extractor(
    feats: pd.DataFrame,
    feat_names: list[str],
    seq_len: int,
    n_features: int,
) -> Callable[[int, str], Optional[np.ndarray]]:
    """Create a function that extracts one `(seq_len, n_features)` sample."""

    def get_feat_matrix(t_idx: int, sym: str) -> Optional[np.ndarray]:
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


def build_dataset(
    mask: np.ndarray,
    dates: pd.Index,
    syms: list[str],
    get_feat_matrix: Callable[[int, str], Optional[np.ndarray]],
    next_ret: pd.DataFrame,
    daily_vol: pd.DataFrame,
    seq_len: int,
    n_features: int,
    sym_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten panel data into supervised samples across (date, symbol)."""

    x_list, r_list, vol_list, sid_list = [], [], [], []
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
            sid_list.append(sym_to_idx[sym])
    if not x_list:
        return (
            np.empty((0, seq_len, n_features), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.int64),
        )
    return (
        np.stack(x_list),
        np.asarray(r_list, np.float32),
        np.asarray(vol_list, np.float32),
        np.asarray(sid_list, np.int64),
    )


def prepare_sequence_data(prices: pd.DataFrame, cfg: BacktestConfig, seq_len: int) -> SequenceDataBundle:
    """Compute features and helpers shared by training and inference."""

    rets = compute_returns(prices)
    daily_vol = ewma_vol(rets, span=cfg.vol_span)
    feats = make_dmn_features(prices, daily_vol)

    feat_names = sorted({c[0] for c in feats.columns})
    n_features = len(feat_names)
    get_feat_matrix = make_feature_extractor(feats, feat_names, seq_len, n_features)

    return SequenceDataBundle(
        dates=prices.index,
        syms=list(prices.columns),
        sym_to_idx={sym: i for i, sym in enumerate(prices.columns)},
        feat_names=feat_names,
        n_features=n_features,
        seq_len=seq_len,
        next_ret=rets.shift(-1),
        daily_vol=daily_vol,
        sigma_tgt_daily=cfg.sigma_target_annual / math.sqrt(252),
        get_feat_matrix=get_feat_matrix,
    )


def build_retrain_points(dates: pd.Index, retrain_years: int) -> list[pd.Timestamp]:
    """Return retraining boundaries for walk-forward evaluation."""

    start_date = dates[0]
    end_date = dates[-1]
    retrain_points: list[pd.Timestamp] = []
    cur = start_date + pd.DateOffset(years=retrain_years)
    while cur < end_date:
        retrain_points.append(cur)
        cur = cur + pd.DateOffset(years=retrain_years)
    retrain_points.append(end_date + pd.Timedelta(days=1))
    return retrain_points


def fit_model_sharpe(
    model_factory: Callable[..., nn.Module],
    model_kwargs: dict,
    x_train: np.ndarray,
    r_train: np.ndarray,
    v_train: np.ndarray,
    sid_train: np.ndarray,
    sigma_tgt_daily: float,
    lr: float,
    epochs: int,
    batch_size: int,
    turnover_lambda: float,
) -> nn.Module:
    """Train one model with Sharpe loss and optional turnover regularization."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_factory(**model_kwargs).to(device)
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
            sidb = torch.tensor(sid_train[idx], dtype=torch.long, device=device)

            pos = net(xb, sidb)
            ret_cap = pos * (sigma_tgt_daily / (vb + 1e-12)) * rb

            loss = sharpe_loss(ret_cap)
            if turnover_lambda > 0:
                loss = loss + turnover_lambda * pos.abs().mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

    return net


def run_walkforward_positions(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    model_factory: Callable[..., nn.Module],
    model_kwargs: dict,
    retrain_years: int,
    seq_len: int,
    lr: float,
    epochs: int,
    batch_size: int,
    turnover_lambda: float,
    seed: int,
    min_train_samples: int = 2000,
) -> pd.DataFrame:
    """Generic walk-forward engine used by LSTM/xLSTM/VLSTM strategies."""

    set_seed(seed)
    data = prepare_sequence_data(prices, cfg, seq_len=seq_len)
    retrain_points = build_retrain_points(data.dates, retrain_years=retrain_years)
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for k in range(len(retrain_points) - 1):
        t0 = retrain_points[k]
        t1 = retrain_points[k + 1]
        train_mask = data.dates < t0
        test_mask = (data.dates >= t0) & (data.dates < t1)

        if int(train_mask.sum()) < cfg.min_obs:
            continue

        x_train, r_train, v_train, sid_train = build_dataset(
            train_mask,
            data.dates,
            data.syms,
            data.get_feat_matrix,
            data.next_ret,
            data.daily_vol,
            data.seq_len,
            data.n_features,
            data.sym_to_idx,
        )
        if len(r_train) < min_train_samples:
            continue

        resolved_model_kwargs = dict(model_kwargs)
        if resolved_model_kwargs.get("n_features") is None:
            resolved_model_kwargs["n_features"] = data.n_features
        if (
            resolved_model_kwargs.get("use_ticker_embedding", False)
            and resolved_model_kwargs.get("n_tickers") is None
        ):
            resolved_model_kwargs["n_tickers"] = len(data.syms)

        net = fit_model_sharpe(
            model_factory=model_factory,
            model_kwargs=resolved_model_kwargs,
            x_train=x_train,
            r_train=r_train,
            v_train=v_train,
            sid_train=sid_train,
            sigma_tgt_daily=data.sigma_tgt_daily,
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
                t = data.dates[t_idx]
                for sym in data.syms:
                    x = data.get_feat_matrix(t_idx, sym)
                    if x is None:
                        continue
                    xb = torch.tensor(x[None, :, :], device=device)
                    sid = torch.tensor([data.sym_to_idx[sym]], dtype=torch.long, device=device)
                    pos = float(net(xb, sid).cpu().numpy()[0])
                    positions.loc[t, sym] = pos

        positions = positions.ffill()

    return positions.fillna(0.0).clip(-1.0, 1.0)
