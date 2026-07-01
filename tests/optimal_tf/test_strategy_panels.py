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
from optimal_tf.strategies.common import resolve_clean_correlation_at_date, resolve_covariance_at_date  # noqa: E402


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

    def test_supported_strategies_include_current_recipes(self) -> None:
        self.assertIn("NM", supported_strategies())
        self.assertIn("EW", supported_strategies())
        self.assertIn("LLTF", supported_strategies())
        self.assertIn("ARP_AGNOSTIC", supported_strategies())
        self.assertIn("ATF_AGNOSTIC", supported_strategies())
        self.assertNotIn("ToRP0", supported_strategies())
        self.assertNotIn("ToRP1", supported_strategies())
        self.assertNotIn("ToRP2", supported_strategies())
        self.assertNotIn("ToRP3", supported_strategies())

    def test_equal_weight_panel_is_row_normalized(self) -> None:
        panel = compute_weights_panel(self.prices, EstimationConfig(), "EW", long_only=True)

        self.assertTrue(np.allclose(panel.sum(axis=1).to_numpy(), np.ones(len(panel))))
        self.assertTrue((panel >= 0.0).all().all())

    def test_nm_panel_builds_from_covariance_panel(self) -> None:
        with patch("optimal_tf.strategies.common.estimate_clean_covariance_at_date", return_value=self.cov) as mocked:
            compute_weights_panel(self.prices, EstimationConfig(), "NM", long_only=False)
            self.assertTrue(mocked.called)

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

    def test_base_and_effective_weights_match_for_nm(self) -> None:
        with patch("optimal_tf.strategies.common.estimate_clean_covariance_at_date", return_value=self.cov):
            panel = compute_strategy_panel(self.prices, EstimationConfig(), "NM", long_only=False)

        last_date = self.prices.index[-1]
        self.assertTrue(np.allclose(
            panel.base_weights.loc[last_date].to_numpy(),
            panel.effective_weights.loc[last_date].to_numpy(),
        ))
        self.assertAlmostEqual(float(panel.signal_scale.loc[last_date]), 1.0)

    def test_agnostic_recipe_panel_is_available_through_public_allocation_api(self) -> None:
        panel = compute_weights_panel(
            self.prices,
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2),
            "ARP_AGNOSTIC",
            long_only=False,
            covariance_cache={self.prices.index[-1]: self.cov},
        )
        row = panel.loc[self.prices.index[-1]]
        self.assertTrue(np.isfinite(row.to_numpy()).all())
        self.assertAlmostEqual(float(row.abs().sum()), 1.0)

    def test_covariance_cache_reuses_recent_snapshot(self) -> None:
        recent_date = self.prices.index[-2]
        target_date = self.prices.index[-1]

        with patch("optimal_tf.strategies.common.estimate_clean_covariance_at_date") as mocked:
            resolved = resolve_covariance_at_date(
                self.prices,
                EstimationConfig(),
                target_date,
                covariance_cache={recent_date: self.cov},
            )

        mocked.assert_not_called()
        self.assertTrue(resolved.equals(self.cov))

    def test_covariance_cache_does_not_reuse_stale_snapshot(self) -> None:
        stale_cov = pd.DataFrame(
            [[0.09, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.01]],
            index=list("ABC"),
            columns=list("ABC"),
        )
        fresh_cov = pd.DataFrame(
            [[0.01, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.09]],
            index=list("ABC"),
            columns=list("ABC"),
        )
        target_date = pd.Timestamp("2026-01-20")

        with patch("optimal_tf.strategies.common.estimate_clean_covariance_at_date", return_value=fresh_cov) as mocked:
            resolved = resolve_covariance_at_date(
                self.prices,
                EstimationConfig(),
                target_date,
                covariance_cache={pd.Timestamp("2026-01-01"): stale_cov},
            )

        mocked.assert_called_once()
        self.assertTrue(resolved.equals(fresh_cov))

    def test_correlation_cache_does_not_reuse_stale_snapshot(self) -> None:
        stale_corr = pd.DataFrame(
            [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )
        fresh_corr = pd.DataFrame(
            [[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )
        target_date = pd.Timestamp("2026-01-20")

        with patch("optimal_tf.strategies.common.estimate_clean_correlation_at_date", return_value=fresh_corr) as mocked:
            resolved = resolve_clean_correlation_at_date(
                self.prices,
                EstimationConfig(),
                target_date,
                correlation_cache={pd.Timestamp("2026-01-01"): stale_corr},
            )

        mocked.assert_called_once()
        self.assertTrue(resolved.equals(fresh_corr))


if __name__ == "__main__":
    unittest.main()
