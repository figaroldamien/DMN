"""CLI entrypoint for live-style DMN LSTM train/predict workflows.

Examples:
  python src/dmn_live.py train --market cac40 --cutoff-mode year_end_prev
  python src/dmn_live.py train --ticker AAPL --cutoff-mode date --cutoff-date 2025-12-31
  python src/dmn_live.py predict --artifact-path artifacts/dmn/dmn_lstm_20251231.pt --from-date 2026-03-01
"""

from __future__ import annotations

from dmn.cli.live import run


if __name__ == "__main__":
    raise SystemExit(run([ "predict", "--market", "cac40", "--artifact-path", "artifacts/dmn/cac40_20251231.pt", "--from-date", "2026-01-01" ]))
    #raise SystemExit(run([ "predict", "--ticker", "AC.PA", "--artifact-path", "artifacts/dmn/AC.PA_20251231.pt", "--from-date", "2026-01-01" ]))
    
    #raise SystemExit(run([ "train", "--ticker", "AC.PA", "--cutoff-mode", "year_end_prev" ]))
    #raise SystemExit(run([ "train", "--market", "cac40", "--cutoff-mode", "year_end_prev" ]))