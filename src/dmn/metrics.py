from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .config import Perf


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = (equity_curve / peak) - 1.0
    return float(dd.min())


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    downside = np.minimum(returns - mar, 0.0)
    return float(np.sqrt(np.mean(downside ** 2)))


def performance_metrics(daily_returns: pd.Series, daily_turnover: Optional[pd.Series] = None) -> Perf:
    r = daily_returns.dropna()
    if len(r) < 30:
        raise ValueError("Not enough returns to compute metrics.")

    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=0) * math.sqrt(252))
    sharpe = float((r.mean() / (r.std(ddof=0) + 1e-12)) * math.sqrt(252))

    dd = downside_deviation(r)
    sortino = float((r.mean() * 252) / ((dd * math.sqrt(252)) + 1e-12))

    eq = (1.0 + r).cumprod()
    mdd = max_drawdown(eq)
    calmar = float(ann_ret / (abs(mdd) + 1e-12))

    pct_pos = float((r > 0).mean())

    profits = r[r > 0]
    losses = r[r < 0]
    avg_p = float(profits.mean()) if len(profits) else 0.0
    avg_l = float(abs(losses.mean())) if len(losses) else np.nan
    apl = float(avg_p / (avg_l + 1e-12)) if np.isfinite(avg_l) else np.nan

    if daily_turnover is None:
        avg_to = float("nan")
    else:
        avg_to = float(daily_turnover.dropna().mean())

    return Perf(
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        mdd=mdd,
        pct_pos=pct_pos,
        avg_profit_over_avg_loss=apl,
        avg_turnover=avg_to,
    )
