from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_core.risk import (  # noqa: E402
    clean_correlation_matrix,
    correlation_to_covariance,
    covariance_to_correlation,
    estimate_clean_covariance_panel,
    marchenko_pastur_law,
)


class TradingCoreRiskTests(unittest.TestCase):
    def test_covariance_roundtrip(self) -> None:
        corr = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["A", "B"], columns=["A", "B"])
        vol = pd.Series({"A": 0.1, "B": 0.2})
        cov = correlation_to_covariance(corr, vol)
        back = covariance_to_correlation(cov)
        np.testing.assert_allclose(back.to_numpy(), corr.to_numpy())

    def test_clean_correlation_matrix_linear_shrinkage(self) -> None:
        corr = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=["A", "B"], columns=["A", "B"])
        cleaned = clean_correlation_matrix(corr, method="linear_shrinkage", linear_shrinkage=0.2)
        self.assertEqual(list(cleaned.index), ["A", "B"])
        np.testing.assert_allclose(np.diag(cleaned), np.ones(2))

    def test_marchenko_pastur_support_and_pdf(self) -> None:
        law = marchenko_pastur_law(num_assets=25, sample_size=100, variance=1.0)
        self.assertAlmostEqual(law.aspect_ratio, 0.25)
        self.assertAlmostEqual(law.lambda_minus, 0.25)
        self.assertAlmostEqual(law.lambda_plus, 2.25)
        grid = np.array([0.1, law.lambda_minus, 1.0, law.lambda_plus, 3.0])
        pdf = law.pdf(grid)
        self.assertEqual(pdf.shape, grid.shape)
        self.assertEqual(float(pdf[0]), 0.0)
        self.assertEqual(float(pdf[-1]), 0.0)
        self.assertGreater(float(pdf[2]), 0.0)
        quantiles = law.quantile_curve(np.arange(1, 6))
        self.assertEqual(quantiles.shape, (5,))
        self.assertTrue(np.all(np.diff(quantiles) <= 1e-12))
        self.assertGreaterEqual(float(quantiles[0]), float(quantiles[-1]))

    def test_clean_correlation_matrix_rie_reference_uses_optional_package(self) -> None:
        fake_module = types.SimpleNamespace(
            get_rie=lambda data, normalize, max_ones: np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float)
        )
        previous = sys.modules.get("rie_estimator")
        sys.modules["rie_estimator"] = fake_module
        try:
            corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"])
            data = pd.DataFrame({"A": [0.1, 0.2], "B": [0.0, 0.3]})
            cleaned = clean_correlation_matrix(corr, data=data, method="rie_reference")
        finally:
            if previous is None:
                sys.modules.pop("rie_estimator", None)
            else:
                sys.modules["rie_estimator"] = previous
        self.assertAlmostEqual(float(cleaned.loc["A", "A"]), 1.0)
        self.assertAlmostEqual(float(cleaned.loc["A", "B"]), 0.25)

    def test_estimate_clean_covariance_panel_accepts_staggered_universe(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100, 101, 102, 103, 104, 105],
                "B": [100, 99, 101, 100, 102, 103],
                "C": [100, np.nan, 101, np.nan, 102, np.nan],
            },
            index=pd.date_range("2026-01-01", periods=6, freq="B"),
        )

        class Cfg:
            vol_span = 2
            covariance_window = 5
            covariance_alpha = None
            corr_span = None
            covariance_min_periods = 3
            max_abs_return = 1.0
            cleaning_method = "linear_shrinkage"
            linear_shrinkage = 0.1
            rie_bandwidth = 1e-3

        panel = estimate_clean_covariance_panel(prices, Cfg(), target_dates=pd.DatetimeIndex([prices.index[-1]]))
        self.assertIn(prices.index[-1], panel)
        cov = panel[prices.index[-1]]
        self.assertGreaterEqual(len(cov.index), 2)


if __name__ == "__main__":
    unittest.main()
