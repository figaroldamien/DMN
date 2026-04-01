from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.metrics import performance_metrics  # noqa: E402
from optimal_tf.validation import compare_cleaners, validate_estimation_config  # noqa: E402
from optimal_tf.config import EstimationConfig  # noqa: E402
from optimal_tf.config_io import load_config  # noqa: E402


class MetricsAndValidationTests(unittest.TestCase):
    def test_performance_metrics_returns_expected_fields(self) -> None:
        pnl = pd.Series([0.01, -0.02, 0.03, 0.0], index=pd.date_range("2024-01-01", periods=4))
        turnover = pd.Series([0.0, 0.1, 0.2, 0.1], index=pnl.index)

        perf = performance_metrics(pnl, turnover)

        self.assertAlmostEqual(perf.ann_return, pnl.mean() * 252)
        self.assertAlmostEqual(perf.ann_vol, pnl.std() * math.sqrt(252))
        self.assertAlmostEqual(perf.avg_turnover, turnover.mean())
        self.assertLessEqual(perf.mdd, 0.0)

    def test_compare_cleaners_computes_abs_differences(self) -> None:
        reference = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]])
        candidate = pd.DataFrame([[1.0, 0.1], [0.15, 1.0]])

        stats = compare_cleaners(reference, candidate)

        self.assertAlmostEqual(stats["max_abs_diff"], 0.1)
        self.assertAlmostEqual(stats["mean_abs_diff"], 0.0375)

    def test_validate_estimation_config_rejects_min_periods_above_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "covariance_min_periods must be less than or equal to covariance_window"):
            validate_estimation_config(EstimationConfig(covariance_window=60, covariance_min_periods=252))

    def test_load_config_rejects_incoherent_covariance_window(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
vol_span = 60
covariance_window = 60
covariance_min_periods = 252
cleaning_method = "empirical"

[backtest]
sigma_target_annual = 0.15
portfolio_vol_target = false
portfolio_vol_span = 60
cost_bps = 0.0
long_only = false

[allocation]
strategy = "RP"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        with self.assertRaisesRegex(ValueError, "covariance_min_periods must be less than or equal to covariance_window"):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
