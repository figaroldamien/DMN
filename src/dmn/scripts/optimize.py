"""Minimal Python entrypoint for hyperparameter grid search.

Examples:
  python -m dmn.scripts.optimize --market cac40 --strategy DMN_LSTM_Sharpe_TurnPen
  python -m dmn.scripts.optimize --ticker AAPL --strategy VLSTM_Sharpe

All other optimization and backtest parameters are loaded from the TOML config.
"""

from __future__ import annotations

from dmn.cli.optimize_cli import run


if __name__ == "__main__":
    raise SystemExit(run())
