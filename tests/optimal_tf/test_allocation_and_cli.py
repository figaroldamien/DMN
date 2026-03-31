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

from optimal_tf.allocation import compute_portfolio_weights_at_date, resolve_allocation_date  # noqa: E402
from optimal_tf.cli.main import run  # noqa: E402
from optimal_tf.config import EstimationConfig  # noqa: E402


class AllocationAndCliTests(unittest.TestCase):
    def test_resolve_allocation_date_uses_latest_available_before_target(self) -> None:
        index = pd.DatetimeIndex(["2026-03-24", "2026-03-25", "2026-03-27"])

        resolved = resolve_allocation_date(index, "2026-03-26")

        self.assertEqual(resolved, pd.Timestamp("2026-03-25"))

    def test_compute_portfolio_weights_at_date_returns_requested_snapshot(self) -> None:
        prices = pd.DataFrame(
            {"A": [1.0, 1.1, 1.2], "B": [1.0, 0.9, 0.95]},
            index=pd.date_range("2026-03-24", periods=3),
        )
        with patch(
            "optimal_tf.allocation.compute_strategy_state_at_date",
            return_value=type(
                "State",
                (),
                {
                    "base_weights": pd.Series({"A": 0.7, "B": 0.3}),
                    "signal_scale": 1.0,
                    "effective_weights": pd.Series({"A": 0.7, "B": 0.3}),
                },
            )(),
        ):
            date, snapshot = compute_portfolio_weights_at_date(prices, EstimationConfig(), "RP", as_of_date="2026-03-26")

        self.assertEqual(date, pd.Timestamp("2026-03-26"))
        self.assertAlmostEqual(float(snapshot["A"]), 0.7)
        self.assertAlmostEqual(float(snapshot["B"]), 0.3)

    def test_cli_prints_weights_for_mocked_data(self) -> None:
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
portfolio_vol_target = true
portfolio_vol_span = 60
cost_bps = 0.0
long_only = false

[allocation]
strategy = "RP"
date = "2026-03-27"
"""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tmp:
            tmp.write(config_text)
            path = tmp.name

        prices = pd.DataFrame({"A": [100.0, 101.0], "B": [100.0, 99.0]}, index=pd.date_range("2026-03-26", periods=2))
        output = io.StringIO()

        with patch("optimal_tf.cli.main.load_prices_for_universe", return_value=prices):
            with patch(
                "optimal_tf.cli.main.compute_portfolio_strategy_state_at_date",
                return_value=(
                    pd.Timestamp("2026-03-27"),
                    type(
                        "State",
                        (),
                        {
                            "base_weights": pd.Series({"A": 0.6, "B": 0.4}),
                            "signal_scale": 1.0,
                            "effective_weights": pd.Series({"A": 0.6, "B": 0.4}),
                        },
                    )(),
                ),
            ):
                with redirect_stdout(output):
                    exit_code = run(["--config", path])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("strategy: RP", rendered)
        self.assertIn("allocation_date: 2026-03-27", rendered)
        self.assertIn("execution_time_seconds:", rendered)
        self.assertIn("A", rendered)
        self.assertIn("0.600000", rendered)

    def test_cli_writes_csv_and_json_outputs(self) -> None:
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
portfolio_vol_target = true
portfolio_vol_span = 60
cost_bps = 0.0
long_only = false

[allocation]
strategy = "RP"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            csv_path = Path(tmpdir) / "weights.csv"
            json_path = Path(tmpdir) / "weights.json"
            config_path.write_text(config_text, encoding="utf-8")

            prices = pd.DataFrame({"A": [100.0, 101.0], "B": [100.0, 99.0]}, index=pd.date_range("2026-03-26", periods=2))
            with patch("optimal_tf.cli.main.load_prices_for_universe", return_value=prices):
                with patch(
                    "optimal_tf.cli.main.compute_portfolio_strategy_state_at_date",
                    return_value=(
                        pd.Timestamp("2026-03-27"),
                        type(
                            "State",
                            (),
                            {
                                "base_weights": pd.Series({"A": 0.6, "B": 0.4}),
                                "signal_scale": 1.0,
                                "effective_weights": pd.Series({"A": 0.6, "B": 0.4}),
                            },
                        )(),
                    ),
                ):
                    exit_code = run(
                        [
                            "--config",
                            str(config_path),
                            "--output-csv",
                            str(csv_path),
                            "--output-json",
                            str(json_path),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("ticker,weight", csv_text)
            self.assertIn("A,0.6", csv_text)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["strategy"], "RP")
            self.assertEqual(payload["universe"], "test")
            self.assertEqual(payload["signal_scale"], 1.0)
            self.assertAlmostEqual(payload["weights"]["A"], 0.6)


if __name__ == "__main__":
    unittest.main()
