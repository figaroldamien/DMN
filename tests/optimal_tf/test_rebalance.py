from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.rebalance import resolve_rebalance_dates  # noqa: E402


class RebalanceTests(unittest.TestCase):
    def test_daily_rebalance_returns_all_dates(self) -> None:
        index = pd.date_range("2026-01-01", periods=4, freq="B")
        resolved = resolve_rebalance_dates(index, "daily")
        self.assertTrue(index.equals(resolved))

    def test_monthly_rebalance_uses_last_available_market_date(self) -> None:
        index = pd.DatetimeIndex(["2026-01-05", "2026-01-29", "2026-02-03", "2026-02-27"])
        resolved = resolve_rebalance_dates(index, "monthly")
        expected = pd.DatetimeIndex(["2026-01-29", "2026-02-27"])
        self.assertTrue(expected.equals(resolved))


if __name__ == "__main__":
    unittest.main()
