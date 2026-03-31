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

from optimal_tf.allocation import compute_strategy_panel, compute_weights_panel, supported_strategies  # noqa: E402
from optimal_tf.config import EstimationConfig  # noqa: E402
from optimal_tf.portfolios import (  # noqa: E402
    risk_parity_weights_from_cov_with_tilt,
    trend_on_risk_parity_weights_from_cov_and_factor_signal,
    trend_on_risk_parity_weights_from_cov_and_signal,
)


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
        self.assertIn("LLTF", supported_strategies())
        self.assertIn("ToRP0", supported_strategies())
        self.assertIn("ToRP1", supported_strategies())
        self.assertIn("ToRP2", supported_strategies())
        self.assertIn("ToRP3", supported_strategies())

    def test_equal_weight_panel_is_row_normalized(self) -> None:
        panel = compute_weights_panel(self.prices, EstimationConfig(), "EW", long_only=True)

        self.assertTrue(np.allclose(panel.sum(axis=1).to_numpy(), np.ones(len(panel))))
        self.assertTrue((panel >= 0.0).all().all())

    def test_nm_panel_builds_from_covariance_panel(self) -> None:
        with patch("optimal_tf.allocation.estimate_clean_covariance_at_date", return_value=self.cov) as mocked:
            compute_weights_panel(self.prices, EstimationConfig(), "NM", long_only=False)
            self.assertTrue(mocked.called)

    def test_torp0_panel_produces_finite_weights(self) -> None:
        cov_panel = {self.prices.index[-1]: self.cov}
        with patch("optimal_tf.allocation.estimate_clean_covariance_panel", return_value=cov_panel):
            with patch("optimal_tf.allocation.estimate_clean_covariance_at_date", return_value=self.cov):
                panel = compute_weights_panel(self.prices, EstimationConfig(vol_span=2, trend_span=2), "ToRP0", long_only=False)

        row = panel.loc[self.prices.index[-1]]
        self.assertTrue(np.isfinite(row.to_numpy()).all())
        self.assertAlmostEqual(float(row.abs().sum()), 1.0)

    def test_torp1_panel_produces_finite_weights(self) -> None:
        cov_panel = {self.prices.index[-1]: self.cov}
        with patch("optimal_tf.allocation.estimate_clean_covariance_panel", return_value=cov_panel):
            with patch("optimal_tf.allocation.estimate_clean_covariance_at_date", return_value=self.cov):
                panel = compute_weights_panel(self.prices, EstimationConfig(vol_span=2, trend_span=2), "ToRP1", long_only=False)

        row = panel.loc[self.prices.index[-1]]
        self.assertTrue(np.isfinite(row.to_numpy()).all())
        self.assertAlmostEqual(float(row.abs().sum()), 1.0)

    def test_torp2_panel_produces_finite_weights(self) -> None:
        cov_panel = {self.prices.index[-1]: self.cov}
        with patch("optimal_tf.allocation.estimate_clean_covariance_panel", return_value=cov_panel):
            with patch("optimal_tf.allocation.estimate_clean_covariance_at_date", return_value=self.cov):
                panel = compute_weights_panel(self.prices, EstimationConfig(vol_span=2, trend_span=2), "ToRP2", long_only=False)

        row = panel.loc[self.prices.index[-1]]
        self.assertTrue(np.isfinite(row.to_numpy()).all())
        self.assertAlmostEqual(float(row.abs().sum()), 1.0)

    def test_torp3_panel_preserves_signal_amplitude(self) -> None:
        cov_panel = {self.prices.index[-1]: self.cov}
        with patch("optimal_tf.allocation.estimate_clean_covariance_panel", return_value=cov_panel):
            panel = compute_strategy_panel(self.prices, EstimationConfig(vol_span=2, trend_span=2), "ToRP3", long_only=False)

        signal = float(panel.signal_scale.loc[self.prices.index[-1]])
        effective = panel.effective_weights.loc[self.prices.index[-1]]
        base = panel.base_weights.loc[self.prices.index[-1]]
        self.assertTrue(np.isfinite(signal))
        self.assertTrue(np.isfinite(effective.to_numpy()).all())
        self.assertTrue(np.allclose(effective.to_numpy(), (base * signal).to_numpy()))

    def test_torp3_signal_gain_scales_effective_weights(self) -> None:
        cov_panel = {self.prices.index[-1]: self.cov}
        with patch("optimal_tf.allocation.estimate_clean_covariance_panel", return_value=cov_panel):
            panel_1 = compute_strategy_panel(
                self.prices,
                EstimationConfig(vol_span=2, trend_span=2, torp_signal_gain=1.0),
                "ToRP3",
                long_only=False,
            )
            panel_2 = compute_strategy_panel(
                self.prices,
                EstimationConfig(vol_span=2, trend_span=2, torp_signal_gain=2.0),
                "ToRP3",
                long_only=False,
            )

        signal_1 = float(panel_1.signal_scale.loc[self.prices.index[-1]])
        signal_2 = float(panel_2.signal_scale.loc[self.prices.index[-1]])
        self.assertAlmostEqual(signal_2, 2.0 * signal_1)

    def test_lltf_panel_produces_finite_normalized_weights(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.5, 101.8, 103.0, 104.2],
                "B": [100.0, 99.5, 100.8, 101.0, 100.7, 101.4],
                "C": [100.0, 100.3, 99.9, 100.4, 100.9, 101.2],
            },
            index=pd.date_range("2026-01-01", periods=6, freq="B"),
        )

        panel = compute_weights_panel(
            prices,
            EstimationConfig(
                vol_span=2,
                trend_span=2,
                covariance_alpha=0.5,
                covariance_min_periods=2,
                lltf_l2_reg=1e-3,
            ),
            "LLTF",
            long_only=False,
        )

        row = panel.loc[prices.index[-1]]
        self.assertTrue(np.isfinite(row.to_numpy()).all())
        self.assertAlmostEqual(float(row.abs().sum()), 1.0)

    def test_torp_uses_projected_rp_factor_signal(self) -> None:
        signal = pd.Series({"A": 2.0, "B": -1.0, "C": 0.5})

        weights = trend_on_risk_parity_weights_from_cov_and_signal(self.cov, signal, long_only=False)
        rp = pd.Series(
            [1.0 / 0.2, 1.0 / 0.3, 1.0 / 0.4],
            index=list("ABC"),
            dtype=float,
        )
        rp = rp / rp.abs().sum()
        projected_signal = float(signal @ rp)
        expected = np.sign(projected_signal) * rp

        self.assertTrue(np.allclose(weights.loc[list("ABC")].to_numpy(), expected.to_numpy()))

    def test_torp_long_only_goes_flat_when_projected_rp_signal_is_negative(self) -> None:
        signal = pd.Series({"A": -2.0, "B": -1.0, "C": -0.5})

        weights = trend_on_risk_parity_weights_from_cov_and_signal(self.cov, signal, long_only=True)

        self.assertTrue(np.allclose(weights.to_numpy(), np.zeros(len(weights))))

    def test_torp2_factor_signal_applies_to_tilted_rp_portfolio(self) -> None:
        tilt = pd.Series({"A": 1.0, "B": 0.0, "C": 1.0})

        weights = trend_on_risk_parity_weights_from_cov_and_factor_signal(
            self.cov,
            factor_signal=0.7,
            tilt=tilt,
            long_only=False,
        )
        expected = risk_parity_weights_from_cov_with_tilt(self.cov, tilt)

        self.assertTrue(np.allclose(weights.loc[list("ABC")].to_numpy(), expected.to_numpy()))
        self.assertAlmostEqual(float(weights.loc["B"]), 0.0)


if __name__ == "__main__":
    unittest.main()
