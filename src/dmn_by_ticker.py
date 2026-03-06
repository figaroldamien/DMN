"""CLI entrypoint for per-ticker strategy evaluation.

Example:
  python src/dmn_by_ticker.py --market cac40 --strategy LongOnly
  python src/dmn_by_ticker.py --market table8_all --strategy DMN_LSTM_Sharpe_TurnPen
  python src/dmn_by_ticker.py --ticker AAPL --strategy MACD_Baz
"""

from __future__ import annotations

from dmn.cli.by_ticker import run


if __name__ == "__main__":
    #raise SystemExit(run(["--market", "cac40", "--strategy", "LongOnly"]))
    raise SystemExit(run(["--market", "cac40", "--strategy", "DMN_LSTM_Sharpe_TurnPen"]))
    #raise SystemExit(run(["--ticker", "AAPL", "--strategy", "MACD_Baz"]))