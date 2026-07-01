"""CLI entrypoint for live-style DMN LSTM train/predict workflows.

Examples:
  python -m dmn.scripts.live train --market cac40 --cutoff-mode year_end_prev
  python -m dmn.scripts.live train --ticker AAPL --cutoff-mode date --cutoff-date 2025-12-31
  python -m dmn.scripts.live predict --artifact-path artifacts/dmn/dmn_lstm_20251231.pt --from-date 2026-03-01
"""

from __future__ import annotations

from dmn.cli.live import run


if __name__ == "__main__":
    raise SystemExit(run())
