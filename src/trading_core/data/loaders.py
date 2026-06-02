from __future__ import annotations

from typing import Iterable

import pandas as pd

from market_tickers_data.prices import load_prices_for_universe as _load_prices_for_universe
from market_tickers_data.prices import load_prices_yf as _load_prices_yf


def load_prices_yf(
    tickers: Iterable[str],
    start: str = '2000-01-01',
    *,
    refresh_policy: str = 'auto',
) -> pd.DataFrame:
    return _load_prices_yf(tickers, start=start, refresh_policy=refresh_policy)


def load_prices_for_universe(
    universe: str,
    start: str = '2000-01-01',
    *,
    refresh_policy: str = 'auto',
) -> pd.DataFrame:
    return _load_prices_for_universe(universe, start=start, refresh_policy=refresh_policy)
