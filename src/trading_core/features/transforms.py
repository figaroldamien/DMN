from __future__ import annotations

import pandas as pd


def normalize_returns_by_vol(returns: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    return returns / (vol + 1e-12)

