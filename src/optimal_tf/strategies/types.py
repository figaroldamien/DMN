from __future__ import annotations

"""Small dataclasses shared by the strategy layer.

The strategy package uses two levels of output:
- `StrategyState` for one rebalance date
- `StrategyPanel` for a whole sequence of rebalance dates

Keeping these as explicit dataclasses makes the contract between strategy code,
services, and dashboard views much easier to reason about than passing around
loosely structured tuples or dictionaries.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyState:
    """State of one strategy on one allocation date.

    Fields:
    - `base_weights`: structural allocation before timing overlays
    - `signal_scale`: scalar timing or amplitude indicator when relevant
    - `effective_weights`: final portfolio actually traded
    """

    base_weights: pd.Series
    signal_scale: float
    effective_weights: pd.Series


@dataclass(frozen=True)
class StrategyPanel:
    """Time-aligned panel version of `StrategyState`.

    Each field is the stacked time-series counterpart of the single-date state:
    - `base_weights`: one row per rebalance date
    - `signal_scale`: one scalar per rebalance date
    - `effective_weights`: the portfolio path actually held over time
    """

    base_weights: pd.DataFrame
    signal_scale: pd.Series
    effective_weights: pd.DataFrame
