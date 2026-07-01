from __future__ import annotations

import io
import json
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

from optimal_tf.cli.compare import run as run_compare  # noqa: E402
from optimal_tf.evaluation import EvaluationResult  # noqa: E402
from optimal_tf.metrics import EvaluationSummary  # noqa: E402


class CompareCliTests(unittest.TestCase):
    def _config_text(self) -> str:
        return """
[universe]
name = "test"
start = "2020-01-01"

[estimation]
vol_span = 60
covariance_alpha = 0.02
covariance_min_periods = 20
cleaning_method = "empirical"
trend_alpha = 0.05
lltf_l2_reg = 0.0001

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

    def _build_result(self, prices: pd.DataFrame, scale: float) -> EvaluationResult:
        summary = EvaluationSummary(
            total_return=0.1 * scale,
            ann_return=0.12 * scale,
            cagr=0.11 * scale,
            ann_vol=0.15,
            sharpe=0.8 * scale,
            sortino=1.0 * scale,
            skewness=0.1,
            mar=2.2 * scale,
            mdd=-0.05,
            avg_turnover=0.2,
            annualized_turnover=10.0,
            total_cost=0.001,
            annualized_cost=0.01,
            pct_positive_days=0.55,
            num_days=3,
            num_rebalances=1,
            avg_turnover_per_rebalance=1.0,
            avg_cost_per_rebalance=0.001,
            total_return_gross=0.101 * scale,
            total_return_cost_drag=0.001,
        )
        idx = [pd.Timestamp("2026-01-30")]
        daily_idx = prices.index
        return EvaluationResult(
            summary=summary,
            weights_by_rebalance=pd.DataFrame({"A": [0.6], "B": [0.4]}, index=idx),
            daily_returns_gross=pd.Series([0.0, 0.01 * scale, -0.005], index=daily_idx),
            daily_returns_net=pd.Series([0.0, 0.009 * scale, -0.006], index=daily_idx),
            turnover_by_rebalance=pd.Series([1.0], index=idx),
            costs_by_rebalance=pd.Series([0.001], index=idx),
            holding_period_returns_gross=pd.Series([0.01 * scale], index=idx),
            holding_period_returns_net=pd.Series([0.009 * scale], index=idx),
            base_weights_by_rebalance=pd.DataFrame({"A": [0.6], "B": [0.4]}, index=idx),
            effective_weights_by_rebalance=pd.DataFrame({"A": [0.6], "B": [0.4]}, index=idx),
            signal_scale_by_rebalance=pd.Series([1.0], index=idx),
        )

    def test_compare_cli_writes_mvp_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            outdir = Path(tmpdir) / "compare"
            config_path.write_text(self._config_text(), encoding="utf-8")
            prices = pd.DataFrame(
                {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 101.0]},
                index=pd.date_range("2026-01-30", periods=3, freq="B"),
            )

            output = io.StringIO()
            with patch("optimal_tf.cli.compare.load_prices_for_universe", return_value=prices):
                with patch(
                    "optimal_tf.cli.compare.compare_strategies",
                    return_value=type(
                        "Comparison",
                        (),
                        {
                            "strategy_results": {"RP": self._build_result(prices, 1.0), "ARP": self._build_result(prices, 1.1)},
                            "summary_table": pd.DataFrame(
                                [
                                    {"strategy": "ARP", "total_return": 0.11, "total_return_gross": 0.111, "total_return_cost_drag": 0.001, "ann_return": 0.132, "ann_vol": 0.15, "sharpe": 0.88, "mdd": -0.05, "avg_turnover": 0.2, "avg_turnover_per_rebalance": 1.0, "annualized_turnover": 10.0, "total_cost": 0.001, "avg_cost_per_rebalance": 0.001, "annualized_cost": 0.01, "pct_positive_days": 0.55, "num_days": 3, "num_rebalances": 1},
                                    {"strategy": "RP", "total_return": 0.10, "total_return_gross": 0.101, "total_return_cost_drag": 0.001, "ann_return": 0.120, "ann_vol": 0.15, "sharpe": 0.80, "mdd": -0.05, "avg_turnover": 0.2, "avg_turnover_per_rebalance": 1.0, "annualized_turnover": 10.0, "total_cost": 0.001, "avg_cost_per_rebalance": 0.001, "annualized_cost": 0.01, "pct_positive_days": 0.55, "num_days": 3, "num_rebalances": 1},
                                ]
                            ),
                            "nav_comparison": pd.DataFrame({"RP": [1.0, 1.01], "ARP": [1.0, 1.02]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                            "drawdown_comparison": pd.DataFrame({"RP": [0.0, -0.01], "ARP": [0.0, -0.005]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                        },
                    )(),
                ):
                    with redirect_stdout(output):
                        exit_code = run_compare(
                            [
                                "--config",
                                str(config_path),
                                "--strategies",
                                "RP,ARP",
                                "--output-dir",
                                str(outdir),
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertTrue((outdir / "manifest.json").exists())
            self.assertTrue((outdir / "inputs.json").exists())
            self.assertTrue((outdir / "comparison" / "summary_table.csv").exists())
            self.assertTrue((outdir / "comparison" / "nav_comparison.csv").exists())
            self.assertTrue((outdir / "comparison" / "drawdown_comparison.csv").exists())
            self.assertTrue((outdir / "comparison" / "plots" / "nav_comparison.png").exists())
            self.assertTrue((outdir / "comparison" / "plots" / "drawdown_comparison.png").exists())
            self.assertTrue((outdir / "strategies" / "RP" / "summary.json").exists())
            rendered = output.getvalue()
            self.assertIn("strategies: RP, ARP", rendered)
            self.assertIn("execution_time_seconds:", rendered)

            manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["strategies"], ["RP", "ARP"])

    def test_compare_cli_cleans_output_dir_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            outdir = Path(tmpdir) / "compare"
            stale_file = outdir / "stale.txt"
            outdir.mkdir(parents=True, exist_ok=True)
            stale_file.write_text("old", encoding="utf-8")
            config_path.write_text(self._config_text(), encoding="utf-8")
            prices = pd.DataFrame(
                {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 101.0]},
                index=pd.date_range("2026-01-30", periods=3, freq="B"),
            )

            with patch("optimal_tf.cli.compare.load_prices_for_universe", return_value=prices):
                with patch(
                    "optimal_tf.cli.compare.compare_strategies",
                    return_value=type(
                        "Comparison",
                        (),
                        {
                            "strategy_results": {"RP": self._build_result(prices, 1.0)},
                            "summary_table": pd.DataFrame(
                                [{"strategy": "RP", "total_return": 0.10, "total_return_gross": 0.101, "total_return_cost_drag": 0.001, "ann_return": 0.120, "ann_vol": 0.15, "sharpe": 0.80, "mdd": -0.05, "avg_turnover": 0.2, "avg_turnover_per_rebalance": 1.0, "annualized_turnover": 10.0, "total_cost": 0.001, "avg_cost_per_rebalance": 0.001, "annualized_cost": 0.01, "pct_positive_days": 0.55, "num_days": 3, "num_rebalances": 1}]
                            ),
                            "nav_comparison": pd.DataFrame({"RP": [1.0, 1.01]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                            "drawdown_comparison": pd.DataFrame({"RP": [0.0, -0.01]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                        },
                    )(),
                ):
                    exit_code = run_compare(
                        [
                            "--config",
                            str(config_path),
                            "--strategies",
                            "RP",
                            "--output-dir",
                            str(outdir),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertFalse(stale_file.exists())

    def test_compare_cli_keeps_output_dir_with_no_clean_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            outdir = Path(tmpdir) / "compare"
            stale_file = outdir / "stale.txt"
            outdir.mkdir(parents=True, exist_ok=True)
            stale_file.write_text("old", encoding="utf-8")
            config_path.write_text(self._config_text(), encoding="utf-8")
            prices = pd.DataFrame(
                {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 101.0]},
                index=pd.date_range("2026-01-30", periods=3, freq="B"),
            )

            with patch("optimal_tf.cli.compare.load_prices_for_universe", return_value=prices):
                with patch(
                    "optimal_tf.cli.compare.compare_strategies",
                    return_value=type(
                        "Comparison",
                        (),
                        {
                            "strategy_results": {"RP": self._build_result(prices, 1.0)},
                            "summary_table": pd.DataFrame(
                                [{"strategy": "RP", "total_return": 0.10, "total_return_gross": 0.101, "total_return_cost_drag": 0.001, "ann_return": 0.120, "ann_vol": 0.15, "sharpe": 0.80, "mdd": -0.05, "avg_turnover": 0.2, "avg_turnover_per_rebalance": 1.0, "annualized_turnover": 10.0, "total_cost": 0.001, "avg_cost_per_rebalance": 0.001, "annualized_cost": 0.01, "pct_positive_days": 0.55, "num_days": 3, "num_rebalances": 1}]
                            ),
                            "nav_comparison": pd.DataFrame({"RP": [1.0, 1.01]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                            "drawdown_comparison": pd.DataFrame({"RP": [0.0, -0.01]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                        },
                    )(),
                ):
                    exit_code = run_compare(
                        [
                            "--config",
                            str(config_path),
                            "--strategies",
                            "RP",
                            "--output-dir",
                            str(outdir),
                            "--no-clean-output-dir",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertTrue(stale_file.exists())

    def test_compare_cli_uses_output_section_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            outdir = Path(tmpdir) / "compare_from_config"
            config_text = self._config_text() + f"""

[output]
compare_dir = "{outdir}"
compare_clean_dir = false
compare_plot = false
"""
            config_path.write_text(config_text, encoding="utf-8")
            prices = pd.DataFrame(
                {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 101.0]},
                index=pd.date_range("2026-01-30", periods=3, freq="B"),
            )

            with patch("optimal_tf.cli.compare.load_prices_for_universe", return_value=prices):
                with patch(
                    "optimal_tf.cli.compare.compare_strategies",
                    return_value=type(
                        "Comparison",
                        (),
                        {
                            "strategy_results": {"RP": self._build_result(prices, 1.0)},
                            "summary_table": pd.DataFrame(
                                [{"strategy": "RP", "total_return": 0.10, "total_return_gross": 0.101, "total_return_cost_drag": 0.001, "ann_return": 0.120, "ann_vol": 0.15, "sharpe": 0.80, "mdd": -0.05, "avg_turnover": 0.2, "avg_turnover_per_rebalance": 1.0, "annualized_turnover": 10.0, "total_cost": 0.001, "avg_cost_per_rebalance": 0.001, "annualized_cost": 0.01, "pct_positive_days": 0.55, "num_days": 3, "num_rebalances": 1}]
                            ),
                            "nav_comparison": pd.DataFrame({"RP": [1.0, 1.01]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                            "drawdown_comparison": pd.DataFrame({"RP": [0.0, -0.01]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                        },
                    )(),
                ):
                    exit_code = run_compare(
                        [
                            "--config",
                            str(config_path),
                            "--strategies",
                            "RP",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertTrue((outdir / "manifest.json").exists())
            self.assertFalse((outdir / "comparison" / "plots" / "nav_comparison.png").exists())

    def test_compare_cli_uses_strategies_from_compare_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            outdir = Path(tmpdir) / "compare_from_config"
            config_text = self._config_text() + f"""

[compare]
strategies = ["RP", "ARP"]

[output]
compare_dir = "{outdir}"
compare_clean_dir = true
compare_plot = false
"""
            config_path.write_text(config_text, encoding="utf-8")
            prices = pd.DataFrame(
                {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 101.0]},
                index=pd.date_range("2026-01-30", periods=3, freq="B"),
            )

            captured = {}

            def fake_compare(prices_arg, estimation_arg, backtest_arg, evaluation_arg, strategies_arg):
                captured["strategies"] = strategies_arg
                return type(
                    "Comparison",
                    (),
                    {
                        "strategy_results": {"RP": self._build_result(prices, 1.0), "ARP": self._build_result(prices, 1.1)},
                        "summary_table": pd.DataFrame(
                            [
                                {"strategy": "ARP", "total_return": 0.11, "total_return_gross": 0.111, "total_return_cost_drag": 0.001, "ann_return": 0.132, "ann_vol": 0.15, "sharpe": 0.88, "mdd": -0.05, "avg_turnover": 0.2, "avg_turnover_per_rebalance": 1.0, "annualized_turnover": 10.0, "total_cost": 0.001, "avg_cost_per_rebalance": 0.001, "annualized_cost": 0.01, "pct_positive_days": 0.55, "num_days": 3, "num_rebalances": 1},
                                {"strategy": "RP", "total_return": 0.10, "total_return_gross": 0.101, "total_return_cost_drag": 0.001, "ann_return": 0.120, "ann_vol": 0.15, "sharpe": 0.80, "mdd": -0.05, "avg_turnover": 0.2, "avg_turnover_per_rebalance": 1.0, "annualized_turnover": 10.0, "total_cost": 0.001, "avg_cost_per_rebalance": 0.001, "annualized_cost": 0.01, "pct_positive_days": 0.55, "num_days": 3, "num_rebalances": 1},
                            ]
                        ),
                        "nav_comparison": pd.DataFrame({"RP": [1.0, 1.01], "ARP": [1.0, 1.02]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                        "drawdown_comparison": pd.DataFrame({"RP": [0.0, -0.01], "ARP": [0.0, -0.005]}, index=pd.date_range("2026-01-30", periods=2, freq="B")),
                    },
                )()

            with patch("optimal_tf.cli.compare.load_prices_for_universe", return_value=prices):
                with patch("optimal_tf.cli.compare.compare_strategies", side_effect=fake_compare):
                    exit_code = run_compare(["--config", str(config_path)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["strategies"], ["RP", "ARP"])


if __name__ == "__main__":
    unittest.main()
