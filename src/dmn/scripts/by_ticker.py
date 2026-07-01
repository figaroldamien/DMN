"""CLI entrypoint for per-ticker strategy evaluation.

Examples:
  python -m dmn.scripts.by_ticker --market cac40 --strategy LongOnly
  python -m dmn.scripts.by_ticker --market table8_all --strategy DMN_LSTM_Sharpe_TurnPen
  python -m dmn.scripts.by_ticker --ticker AAPL --strategy MACD_Baz
"""

from __future__ import annotations

from dmn.cli.by_ticker import run


if __name__ == "__main__":
    raise SystemExit(run())
