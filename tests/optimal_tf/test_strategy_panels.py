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

from optimal_tf.allocation import compute_weights_panel, supported_strategies  # noqa: E402
from optimal_tf.config import EstimationConfig  # noqa: E402


class StrategyPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0],
                "B": [100.0, 99.0, 100.0, 101.0],
                "C": [100.0, 100.5, 100.0, 99.5],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="B"),
        )
        self.cov = pd.DataFrame(
            [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]],
            index=list("ABC"),
            columns=list("ABC"),
        )

    def test_supported_strategies_include_new_recipes(self) -> None:
        self.assertIn("NM", supported_strategies())
        self.assertIn("EW", supported_strategies())
        self.assertIn("ToRP", supported_strategies())

    def test_equal_weight_panel_is_row_normalized(self) -> None:
        panel = compute_weights_panel(self.prices, EstimationConfig(), "EW", long_only=True)

        self.assertTrue(np.allclose(panel.sum(axis=1).to_numpy(), np.ones(len(panel))))
        self.assertTrue((panel >= 0.0).all().all())

    def test_nm_panel_builds_from_covariance_panel(self) -> None:
        with patch("optimal_tf.allocation.build_weight_panel") as mocked:
            mocked.return_value = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
            compute_weights_panel(self.prices, EstimationConfig(), "NM", long_only=False)
            mocked.assert_called_once()

    def test_torp_panel_produces_finite_weights(self) -> None:
        cov_panel = {self.prices.index[-1]: self.cov}
        with patch("optimal_tf.allocation.estimate_clean_covariance_panel", return_value=cov_panel):
            panel = compute_weights_panel(self.prices, EstimationConfig(vol_span=2, trend_span=2), "ToRP", long_only=False)

        row = panel.loc[self.prices.index[-1]]
        self.assertTrue(np.isfinite(row.to_numpy()).all())
        self.assertAlmostEqual(float(row.abs().sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
