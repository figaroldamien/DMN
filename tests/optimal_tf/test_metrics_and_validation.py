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
from trading_core.reporting.metrics import evaluation_metrics  # noqa: E402
from optimal_tf.validation import compare_cleaners, validate_backtest_config, validate_estimation_config  # noqa: E402
from optimal_tf.config import BacktestConfig, EstimationConfig  # noqa: E402
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

    def test_evaluation_metrics_reports_gross_return_and_per_rebalance_averages(self) -> None:
        net = pd.Series([0.0, 0.009, -0.006], index=pd.date_range("2026-01-30", periods=3, freq="B"))
        gross = pd.Series([0.0, 0.010, -0.005], index=net.index)
        turnover = pd.Series([0.0, 1.0, 0.0], index=net.index)
        costs = pd.Series([0.001], index=[pd.Timestamp("2026-01-30")])

        summary = evaluation_metrics(net, turnover, costs, gross_pnl=gross, num_rebalances=1)

        self.assertAlmostEqual(summary.avg_turnover_per_rebalance, 1.0)
        self.assertAlmostEqual(summary.avg_cost_per_rebalance, 0.001)
        self.assertGreater(summary.total_return_gross, summary.total_return)
        self.assertAlmostEqual(summary.total_return_cost_drag, summary.total_return_gross - summary.total_return)
        self.assertGreater(summary.sortino, 0.0)
        self.assertNotEqual(summary.skewness, 0.0)
        self.assertGreater(summary.mar, 0.0)

    def test_validate_estimation_config_rejects_min_periods_above_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "covariance_min_periods must be less than or equal to covariance_window"):
            validate_estimation_config(EstimationConfig(covariance_window=60, covariance_min_periods=252))

    def test_validate_estimation_config_rejects_unknown_cleaning_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "cleaning_method must be one of"):
            validate_estimation_config(EstimationConfig(cleaning_method="unknown"))

    def test_validate_backtest_config_rejects_invalid_weight_smoothing_alpha(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight_smoothing_alpha must be in the interval"):
            validate_backtest_config(BacktestConfig(weight_smoothing_alpha=0.0))

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

    def test_load_config_reads_output_section(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
vol_span = 60
covariance_window = 120
covariance_min_periods = 60
cleaning_method = "empirical"

[allocation]
strategy = "RP"

[output]
allocation_csv = "output/weights.csv"
allocation_json = "output/weights.json"
evaluation_dir = "output/eval"
evaluation_plot = false
compare_dir = "output/compare"
compare_clean_dir = false
compare_plot = false
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        _, _, _, _, _, _, output = load_config(path)

        self.assertEqual(output.allocation_csv, "output/weights.csv")
        self.assertEqual(output.allocation_json, "output/weights.json")
        self.assertEqual(output.evaluation_dir, "output/eval")
        self.assertFalse(output.evaluation_plot)
        self.assertEqual(output.compare_dir, "output/compare")
        self.assertFalse(output.compare_clean_dir)
        self.assertFalse(output.compare_plot)

    def test_load_config_reads_compare_section(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
vol_span = 60
covariance_window = 120
covariance_min_periods = 60
cleaning_method = "empirical"

[compare]
strategies = ["RP", "ARP", "LLTF"]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        _, _, _, _, _, compare, _ = load_config(path)

        self.assertEqual(compare.strategies, ("RP", "ARP", "LLTF"))

    def test_load_config_reads_backtest_weight_smoothing_alpha(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
vol_span = 60
covariance_window = 120
covariance_min_periods = 60
cleaning_method = "empirical"

[backtest]
weight_smoothing_alpha = 0.35
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        _, _, backtest, _, _, _, _ = load_config(path)

        self.assertAlmostEqual(backtest.weight_smoothing_alpha, 0.35)

    def test_load_config_accepts_portfolio_alias_for_backtest(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
vol_span = 60
covariance_window = 120
covariance_min_periods = 60
cleaning_method = "empirical"

[portfolio]
weight_smoothing_alpha = 0.35
cost_bps = 12.0
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        _, _, backtest, _, _, _, _ = load_config(path)

        self.assertAlmostEqual(backtest.weight_smoothing_alpha, 0.35)
        self.assertAlmostEqual(backtest.cost_bps, 12.0)

    def test_load_config_derives_trend_span_from_explicit_alpha(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
trend_alpha = 0.01575
covariance_window = 252
covariance_min_periods = 252
cleaning_method = "empirical"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        _, estimation, _, _, _, _, _ = load_config(path)

        self.assertAlmostEqual(float(estimation.trend_alpha or 0.0), 0.01575)
        self.assertEqual(estimation.trend_span, 126)

    def test_load_config_derives_trend_alpha_from_explicit_span(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
trend_span = 126
covariance_window = 252
covariance_min_periods = 252
cleaning_method = "empirical"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        _, estimation, _, _, _, _, _ = load_config(path)

        self.assertEqual(estimation.trend_span, 126)
        self.assertAlmostEqual(float(estimation.trend_alpha or 0.0), 2.0 / 127.0)


if __name__ == "__main__":
    unittest.main()
