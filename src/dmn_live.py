"""CLI entrypoint for live-style DMN LSTM train/predict workflows.

Examples:
  python src/dmn_live.py train --market cac40 --cutoff-mode year_end_prev
  python src/dmn_live.py train --ticker AAPL --cutoff-mode date --cutoff-date 2025-12-31
  python src/dmn_live.py predict --artifact-path artifacts/dmn/dmn_lstm_20251231.pt --from-date 2026-03-01
"""

from __future__ import annotations

from dmn.cli.live import run


if __name__ == "__main__":
    raise SystemExit(run([ "predict", "--artifact-path", "artifacts/dmn/dmn_lstm_20251231.pt", "--from-date", "2026-01-01" ]))

    #raise SystemExit(run([ "train", "--ticker", "AAPL", "--cutoff-mode", "year_end_prev" ]))