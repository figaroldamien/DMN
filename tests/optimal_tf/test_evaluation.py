from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.cli.evaluate import run as run_evaluate  # noqa: E402
from optimal_tf.config import BacktestConfig, EstimationConfig, EvaluationConfig  # noqa: E402
from optimal_tf.evaluation import _apply_portfolio_vol_target, evaluate_portfolio  # noqa: E402
from optimal_tf.features import sanitize_returns  # noqa: E402
from optimal_tf.reporting import equal_weight_buy_and_hold_benchmark, equal_weight_rebalanced_benchmark  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_evaluate_portfolio_applies_cost_on_rebalance(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0],
                "B": [100.0, 100.0, 100.0, 100.0],
            },
            index=pd.to_datetime(["2026-01-30", "2026-02-02", "2026-02-27", "2026-03-02"]),
        )
        weights_panel = pd.DataFrame(
            {
                "A": [0.0, 1.0, 0.0, 0.0],
                "B": [0.0, 0.0, 1.0, 1.0],
            },
            index=prices.index,
        )
        est_cfg = EstimationConfig()
        bt_cfg = BacktestConfig(cost_bps=10.0, portfolio_vol_target=False, long_only=False)
        eval_cfg = EvaluationConfig(strategy="RP", rebalance_frequency="monthly")

        with patch(
            "optimal_tf.evaluation.compute_strategy_panel",
            return_value=type(
                "Panel",
                (),
                {
                    "base_weights": weights_panel,
                    "effective_weights": weights_panel,
                    "signal_scale": pd.Series(1.0, index=weights_panel.index),
                },
            )(),
        ):
            result = evaluate_portfolio(prices, est_cfg, bt_cfg, eval_cfg)

        self.assertEqual(int(result.summary.num_rebalances), 2)
        self.assertGreater(float(result.costs_by_rebalance.sum()), 0.0)
        self.assertLessEqual(result.daily_returns_net.sum(), result.daily_returns_gross.sum())

    def test_evaluate_cli_prints_summary(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
vol_span = 60
corr_span = 252
corr_min_periods = 252
cleaning_method = "empirical"
linear_shrinkage = 0.0
rie_bandwidth = 0.001
trend_span = 252

[backtest]
sigma_target_annual = 0.15
portfolio_vol_target = false
portfolio_vol_span = 60
cost_bps = 0.0
long_only = false

[allocation]
strategy = "RP"

[evaluation]
strategy = "RP"
rebalance_frequency = "monthly"
evaluation_start = "2026-01-01"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        prices = pd.DataFrame({"A": [100.0, 101.0], "B": [100.0, 99.0]}, index=pd.date_range("2026-01-30", periods=2, freq="B"))
        output = io.StringIO()
        fake_summary = {
            "total_return": 0.1,
            "ann_return": 0.12,
            "ann_vol": 0.15,
            "sharpe": 0.8,
            "mdd": -0.05,
            "avg_turnover": 0.2,
            "annualized_turnover": 10.0,
            "total_cost": 0.001,
            "annualized_cost": 0.01,
            "pct_positive_days": 0.55,
            "num_days": 100,
            "num_rebalances": 5,
        }

        from optimal_tf.evaluation import EvaluationResult
        from optimal_tf.metrics import EvaluationSummary

        result = EvaluationResult(
            summary=EvaluationSummary(**fake_summary),
            weights_by_rebalance=pd.DataFrame({"A": [0.6], "B": [0.4]}, index=[pd.Timestamp("2026-01-30")]),
            daily_returns_gross=pd.Series([0.01], index=[pd.Timestamp("2026-02-02")]),
            daily_returns_net=pd.Series([0.009], index=[pd.Timestamp("2026-02-02")]),
            turnover_by_rebalance=pd.Series([1.0], index=[pd.Timestamp("2026-01-30")]),
            costs_by_rebalance=pd.Series([0.001], index=[pd.Timestamp("2026-01-30")]),
            holding_period_returns_gross=pd.Series([0.01], index=[pd.Timestamp("2026-01-30")]),
            holding_period_returns_net=pd.Series([0.009], index=[pd.Timestamp("2026-01-30")]),
        )

        with patch("optimal_tf.cli.evaluate.load_prices_for_universe", return_value=prices):
            with patch("optimal_tf.cli.evaluate.evaluate_portfolio", return_value=result):
                with redirect_stdout(output):
                    exit_code = run_evaluate(["--config", path])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("rebalance_frequency: monthly", rendered)
        self.assertIn("total_return: 0.1", rendered)

    def test_evaluate_cli_writes_plot_when_output_dir_is_provided(self) -> None:
        config_text = """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
vol_span = 60
corr_span = 252
corr_min_periods = 252
cleaning_method = "empirical"
linear_shrinkage = 0.0
rie_bandwidth = 0.001
trend_span = 252

[backtest]
sigma_target_annual = 0.15
portfolio_vol_target = false
portfolio_vol_span = 60
cost_bps = 0.0
long_only = false

[allocation]
strategy = "RP"

[evaluation]
strategy = "RP"
rebalance_frequency = "monthly"
evaluation_start = "2026-01-01"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            outdir = Path(tmpdir) / "out"
            config_path.write_text(config_text, encoding="utf-8")

            prices = pd.DataFrame(
                {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 101.0]},
                index=pd.date_range("2026-01-30", periods=3, freq="B"),
            )

            from optimal_tf.evaluation import EvaluationResult
            from optimal_tf.metrics import EvaluationSummary

            result = EvaluationResult(
                summary=EvaluationSummary(
                    total_return=0.1,
                    ann_return=0.12,
                    ann_vol=0.15,
                    sharpe=0.8,
                    mdd=-0.05,
                    avg_turnover=0.2,
                    annualized_turnover=10.0,
                    total_cost=0.001,
                    annualized_cost=0.01,
                    pct_positive_days=0.55,
                    num_days=3,
                    num_rebalances=1,
                ),
                weights_by_rebalance=pd.DataFrame({"A": [0.6], "B": [0.4]}, index=[pd.Timestamp("2026-01-30")]),
                daily_returns_gross=pd.Series([0.0, 0.01, -0.005], index=prices.index),
                daily_returns_net=pd.Series([0.0, 0.009, -0.006], index=prices.index),
                turnover_by_rebalance=pd.Series([1.0], index=[pd.Timestamp("2026-01-30")]),
                costs_by_rebalance=pd.Series([0.001], index=[pd.Timestamp("2026-01-30")]),
                holding_period_returns_gross=pd.Series([0.01], index=[pd.Timestamp("2026-01-30")]),
                holding_period_returns_net=pd.Series([0.009], index=[pd.Timestamp("2026-01-30")]),
            )

            with patch("optimal_tf.cli.evaluate.load_prices_for_universe", return_value=prices):
                with patch("optimal_tf.cli.evaluate.evaluate_portfolio", return_value=result):
                    exit_code = run_evaluate(["--config", str(config_path), "--output-dir", str(outdir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((outdir / "performance.png").exists())

    def test_portfolio_vol_targeting_scales_after_enough_history(self) -> None:
        gross = pd.Series(
            [0.01, 0.02, -0.01, 0.015, -0.005],
            index=pd.date_range("2026-01-01", periods=5, freq="B"),
        )
        cfg = BacktestConfig(
            sigma_target_annual=0.10,
            portfolio_vol_target=True,
            portfolio_vol_span=2,
            cost_bps=0.0,
            long_only=False,
        )

        scaled, scale = _apply_portfolio_vol_target(gross, cfg)

        self.assertEqual(len(scaled), len(gross))
        self.assertEqual(len(scale), len(gross))
        self.assertAlmostEqual(float(scale.iloc[0]), 1.0)
        self.assertTrue((scale >= 0.0).all())
        self.assertFalse(scaled.equals(gross))

    def test_sanitize_returns_removes_outliers(self) -> None:
        returns = pd.Series([0.01, 0.5, 5.0, -1.5])
        cleaned = sanitize_returns(returns, max_abs_return=1.0)
        self.assertTrue(pd.isna(cleaned.iloc[2]))
        self.assertTrue(pd.isna(cleaned.iloc[3]))
        self.assertEqual(cleaned.iloc[1], 0.5)

    def test_benchmarks_filter_extreme_returns(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0],
                "B": [100.0, 10000.0, 10100.0],
            },
            index=pd.date_range("2026-01-01", periods=3, freq="B"),
        )
        ew = equal_weight_rebalanced_benchmark(prices, max_abs_return=1.0)
        bh = equal_weight_buy_and_hold_benchmark(prices, max_abs_return=1.0)
        self.assertLess(abs(float(ew.iloc[1])), 0.1)
        self.assertLess(abs(float(bh.iloc[1])), 0.1)


if __name__ == "__main__":
    unittest.main()
