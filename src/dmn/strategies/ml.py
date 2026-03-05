from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..config import BacktestConfig
from ..features import compute_returns, ewma_vol, make_dmn_features


def ml_supervised_positions(
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    model_type: str = "lasso_reg",
    retrain_years: int = 5,
) -> pd.DataFrame:
    if Lasso is None or StandardScaler is None:
        raise ImportError("scikit-learn not installed. pip install scikit-learn")

    rets = compute_returns(prices)
    daily_vol = ewma_vol(rets, span=cfg.vol_span)
    feats = make_dmn_features(prices, daily_vol)

    y_reg = rets.shift(-1) / (daily_vol + 1e-12)
    y_dir = (y_reg > 0).astype(int)

    def panel_to_samples(
        xf: pd.DataFrame,
        y: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[pd.Timestamp, str]]]:
        idx = xf.index.intersection(y.index)
        xf = xf.loc[idx]
        y = y.loc[idx]

        feat_names = sorted({c[0] for c in xf.columns})
        syms = list(y.columns)

        samples = []
        targets = []
        keys = []
        for t in idx:
            row = xf.loc[t]
            for sym in syms:
                x = np.array([row[(fn, sym)] for fn in feat_names], dtype=float)
                yv = float(y.loc[t, sym])
                if np.any(np.isnan(x)) or np.isnan(yv):
                    continue
                samples.append(x)
                targets.append(yv)
                keys.append((t, sym))
        return np.asarray(samples), np.asarray(targets), keys

    dates = prices.index
    start_date = dates[0]
    end_date = dates[-1]
    retrain_points = []
    cur = start_date
    while cur < end_date:
        retrain_points.append(cur)
        cur = cur + pd.DateOffset(years=retrain_years)
    retrain_points.append(end_date + pd.Timedelta(days=1))

    positions = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for k in range(len(retrain_points) - 1):
        t0 = retrain_points[k]
        t1 = retrain_points[k + 1]
        train_slice = dates < t0
        test_slice = (dates >= t0) & (dates < t1)

        if train_slice.sum() < cfg.min_obs:
            continue

        x_train_df = feats.loc[train_slice]
        y_train_df = y_dir.loc[train_slice] if "clf" in model_type else y_reg.loc[train_slice]

        x_train, y_train, _ = panel_to_samples(x_train_df, y_train_df)
        if len(y_train) < 500:
            continue

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)

        if model_type == "lasso_reg":
            model = Lasso(alpha=1e-4, max_iter=5000)
        elif model_type == "mlp_reg":
            if MLPRegressor is None:
                raise ImportError("scikit-learn missing MLPRegressor")
            model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=200, random_state=0)
        elif model_type == "lasso_clf":
            model = Lasso(alpha=1e-4, max_iter=5000)
        else:
            raise ValueError("Unknown model_type")

        model.fit(x_train_s, y_train)

        x_test_df = feats.loc[test_slice]
        x_test, _, keys = panel_to_samples(x_test_df, y_reg.loc[test_slice])
        if len(keys) == 0:
            continue

        pred = model.predict(scaler.transform(x_test))
        if model_type in ("lasso_reg", "mlp_reg"):
            x = np.sign(pred)
        else:
            p = np.clip(pred, 0.0, 1.0)
            x = np.sign(p - 0.5)

        for (t, sym), xv in zip(keys, x):
            positions.loc[t, sym] = float(np.clip(xv, -1.0, 1.0))

    return positions.ffill().fillna(0.0)
