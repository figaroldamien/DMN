from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.backtest import (  # noqa: E402
    _portfolio_returns_from_weights,
    backtest_portfolio,
    build_weight_panel,
)
from optimal_tf.config import BacktestConfig, EstimationConfig  # noqa: E402


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2024-01-01", periods=4)
        self.prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0],
                "B": [100.0, 100.0, 101.0, 102.0],
            },
            index=self.dates,
        )
        self.cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )

    def test_portfolio_returns_use_lagged_weights(self) -> None:
        returns = pd.DataFrame(
            {"A": [0.0, 0.01, 0.02], "B": [0.0, 0.03, -0.01]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        weights = pd.DataFrame(
            {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]},
            index=returns.index,
        )
        cfg = BacktestConfig(portfolio_vol_target=False, cost_bps=0.0)

        pnl, turnover = _portfolio_returns_from_weights(returns, weights, cfg)

        self.assertAlmostEqual(float(pnl.iloc[1]), 0.02)
        self.assertAlmostEqual(float(pnl.iloc[2]), 0.005)
        self.assertAlmostEqual(float(turnover.fillna(0.0).sum()), 0.0)

    def test_build_weight_panel_applies_long_only_projection(self) -> None:
        est_cfg = EstimationConfig()

        with patch("optimal_tf.backtest.estimate_clean_covariance_panel", return_value={self.dates[1]: self.cov}):
            weights = build_weight_panel(
                self.prices,
                est_cfg,
                lambda cov: pd.Series({"A": 0.6, "B": -0.4}),
                long_only=True,
            )

        self.assertTrue((weights.loc[self.dates[1] :] >= 0.0).all().all())
        self.assertAlmostEqual(float(weights.loc[self.dates[1]].sum()), 1.0)
        self.assertAlmostEqual(float(weights.loc[self.dates[1], "B"]), 0.0)

    def test_backtest_portfolio_returns_weight_panel(self) -> None:
        est_cfg = EstimationConfig()
        bt_cfg = BacktestConfig(portfolio_vol_target=False, cost_bps=0.0)

        with patch("optimal_tf.backtest.estimate_clean_covariance_panel", return_value={self.dates[1]: self.cov}):
            pnl, turnover, weights = backtest_portfolio(
                self.prices,
                est_cfg,
                bt_cfg,
                lambda cov: pd.Series({"A": 0.5, "B": 0.5}),
            )

        self.assertListEqual(list(weights.columns), ["A", "B"])
        self.assertEqual(len(pnl), len(self.prices))
        self.assertEqual(len(turnover), len(self.prices))
        self.assertTrue(np.isfinite(weights.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
