from __future__ import annotations

from typing import Dict, List

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def load_prices_yf(tickers: List[str], start: str = "2000-01-01") -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance not installed. pip install yfinance")
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy()
    else:
        px = data.rename(columns={"Close": tickers[0]})[tickers[0]].to_frame()
    return px.dropna(how="all").ffill()


def load_prices_csv(path_by_symbol: Dict[str, str], price_col: str = "Close") -> pd.DataFrame:
    frames = []
    for sym, path in path_by_symbol.items():
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise ValueError(f"{path}: needs a Date column")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        if price_col not in df.columns:
            raise ValueError(f"{path}: missing column {price_col}")
        frames.append(df[[price_col]].rename(columns={price_col: sym}))
    return pd.concat(frames, axis=1).dropna(how="all").ffill()
