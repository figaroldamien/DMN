from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyState:
    base_weights: pd.Series
    signal_scale: float
    effective_weights: pd.Series


@dataclass(frozen=True)
class StrategyPanel:
    base_weights: pd.DataFrame
    signal_scale: pd.Series
    effective_weights: pd.DataFrame
