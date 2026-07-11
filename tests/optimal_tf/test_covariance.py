from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.estimators.covariance import (  # noqa: E402
    correlation_to_covariance,
    covariance_to_correlation,
    make_psd,
)
from optimal_tf.estimators.pipeline import estimate_clean_covariance_panel  # noqa: E402
from optimal_tf.config import EstimationConfig  # noqa: E402
from optimal_tf.estimators.cleaning import clean_correlation_matrix_rich  # noqa: E402
from optimal_tf.estimators.rie import clean_correlation_matrix  # noqa: E402
from optimal_tf.estimators.rie_spectral import (  # noqa: E402
    clean_correlation_matrix_rie_spectral,
    clean_covariance_matrix_rie_spectral,
    clean_matrix_rie_spectral,
)
from optimal_tf.features import alpha_from_span, ewma_cov_frame  # noqa: E402
from trading_core.risk import estimate_clean_correlation_panel  # noqa: E402


class CovarianceEstimatorTests(unittest.TestCase):
    def test_covariance_to_correlation_has_unit_diagonal(self) -> None:
        cov = pd.DataFrame(
            [[4.0, 1.0, 0.5], [1.0, 9.0, 3.0], [0.5, 3.0, 16.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )

        corr = covariance_to_correlation(cov)

        np.testing.assert_allclose(np.diag(corr), np.ones(3))
        self.assertTrue(np.allclose(corr.to_numpy(), corr.to_numpy().T))

    def test_correlation_to_covariance_roundtrip_recovers_diagonal(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.2], [0.2, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        vol = pd.Series({"A": 0.1, "B": 0.2})

        cov = correlation_to_covariance(corr, vol)

        np.testing.assert_allclose(np.diag(cov), np.array([0.01, 0.04]))
        np.testing.assert_allclose(cov.loc["A", "B"], 0.004)

    def test_make_psd_floors_negative_eigenvalues(self) -> None:
        matrix = pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )

        fixed = make_psd(matrix, floor=1e-6)
        eigenvalues = np.linalg.eigvalsh(fixed.to_numpy())

        self.assertGreaterEqual(float(eigenvalues.min()), 1e-6 * 0.999)

    def test_linear_shrinkage_returns_valid_correlation(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.8, -0.3], [0.8, 1.0, 0.1], [-0.3, 0.1, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )

        cleaned = clean_correlation_matrix(corr, method="linear_shrinkage", linear_shrinkage=0.4)

        np.testing.assert_allclose(np.diag(cleaned), np.ones(3))
        self.assertTrue(np.allclose(cleaned.to_numpy(), cleaned.to_numpy().T))

    def test_empirical_rich_api_returns_native_spectral_outputs(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.8, -0.3], [0.8, 1.0, 0.1], [-0.3, 0.1, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )

        result = clean_correlation_matrix_rich(corr, method="empirical", sample_size=252)

        self.assertEqual(result.method, "empirical")
        self.assertEqual(result.spectral_source, "matrix_diagonalization")
        np.testing.assert_allclose(
            result.cleaned.eigenvectors.to_numpy(dtype=float),
            result.empirical.eigenvectors.to_numpy(dtype=float),
        )
        np.testing.assert_allclose(result.cleaned_matrix.to_numpy(dtype=float), result.input_matrix.to_numpy(dtype=float))

    def test_linear_shrinkage_rich_api_keeps_empirical_eigenvectors(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.8, -0.3], [0.8, 1.0, 0.1], [-0.3, 0.1, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )

        result = clean_correlation_matrix_rich(
            corr,
            method="linear_shrinkage",
            sample_size=252,
            linear_shrinkage=0.4,
        )

        self.assertEqual(result.method, "linear_shrinkage")
        np.testing.assert_allclose(
            result.cleaned.eigenvectors.to_numpy(dtype=float),
            result.empirical.eigenvectors.to_numpy(dtype=float),
        )
        expected = ((1.0 - 0.4) * result.empirical.eigenvalues) + 0.4
        np.testing.assert_allclose(result.cleaned.eigenvalues, expected)

    def test_rie_spectral_returns_valid_correlation_matrix(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.6, 0.2], [0.6, 1.0, 0.1], [0.2, 0.1, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )

        cleaned = clean_correlation_matrix(corr, method="rie_spectral", sample_size=252, bandwidth=1e-3)

        np.testing.assert_allclose(np.diag(cleaned), np.ones(3), atol=1e-8)
        self.assertTrue(np.allclose(cleaned.to_numpy(), cleaned.to_numpy().T))
        eigvals = np.linalg.eigvalsh(cleaned.to_numpy())
        self.assertGreaterEqual(float(eigvals.min()), -1e-8)

    def test_rie_spectral_keeps_identity_close_to_identity(self) -> None:
        corr = pd.DataFrame(np.eye(4), index=list("ABCD"), columns=list("ABCD"))

        cleaned = clean_correlation_matrix(corr, method="rie_spectral", sample_size=252, bandwidth=1e-3)

        np.testing.assert_allclose(cleaned.to_numpy(), np.eye(4), atol=1e-3)

    def test_rie_spectral_handles_repeated_eigenvalues_without_warning_or_nan(self) -> None:
        corr = pd.DataFrame(
            np.ones((3, 3)),
            index=list("ABC"),
            columns=list("ABC"),
        )

        cleaned = clean_correlation_matrix(corr, method="rie_spectral", sample_size=252, bandwidth=1e-6)

        self.assertTrue(np.isfinite(cleaned.to_numpy()).all())
        np.testing.assert_allclose(np.diag(cleaned), np.ones(3), atol=1e-8)

    def test_rie_reference_rich_api_returns_spectral_outputs(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.6, 0.2], [0.6, 1.0, 0.1], [0.2, 0.1, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )

        result = clean_correlation_matrix_rie_spectral(corr, sample_size=252)

        self.assertEqual(result.matrix_type, "correlation")
        self.assertEqual(result.input_matrix.shape, (3, 3))
        self.assertEqual(result.empirical.eigenvectors.shape, (3, 3))
        self.assertEqual(result.cleaned.eigenvectors.shape, (3, 3))
        self.assertEqual(len(result.cleaned_eigenvalues_by_input_order), 3)
        self.assertTrue(result.postprocess_applied)
        np.testing.assert_allclose(np.diag(result.cleaned_matrix), np.ones(3), atol=1e-8)

    def test_rie_reference_cleaned_order_matches_sorted_xi_hat(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.8, 0.3], [0.8, 1.0, 0.1], [0.3, 0.1, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )

        result = clean_correlation_matrix_rie_spectral(corr, sample_size=252)
        expected = result.cleaned_eigenvalues_by_input_order[list(result.cleaned.rank_order)]

        np.testing.assert_allclose(result.cleaned.eigenvalues, expected)

    def test_rie_reference_covariance_wrapper_roundtrips_vol_scale(self) -> None:
        corr = pd.DataFrame(
            [[1.0, 0.25], [0.25, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        vol = pd.Series({"A": 0.1, "B": 0.2})
        cov = correlation_to_covariance(corr, vol)

        result = clean_covariance_matrix_rie_spectral(cov, sample_size=252)
        dispatched = clean_matrix_rie_spectral(cov, matrix_type="covariance", sample_size=252)

        self.assertEqual(result.matrix_type, "covariance")
        np.testing.assert_allclose(np.diag(result.cleaned_matrix), np.diag(dispatched.cleaned_matrix))
        np.testing.assert_allclose(result.volatility.to_numpy(dtype=float), vol.to_numpy(dtype=float))

    def test_alpha_from_span_matches_ewm_conversion(self) -> None:
        self.assertAlmostEqual(float(alpha_from_span(199)), 0.01)

    def test_ewma_cov_frame_produces_square_matrices(self) -> None:
        frame = pd.DataFrame(
            {
                "A": [0.01, 0.02, -0.01, 0.015],
                "B": [0.03, 0.01, -0.02, 0.01],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="B"),
        )

        panel = ewma_cov_frame(frame, alpha=0.2, min_periods=2)

        self.assertGreaterEqual(len(panel), 1)
        for _, (cov, sample_size) in panel.items():
            self.assertEqual(list(cov.index), list(cov.columns))
            self.assertTrue(np.allclose(cov.to_numpy(), cov.to_numpy().T))
            self.assertGreaterEqual(sample_size, 1)

    def test_estimate_clean_covariance_panel_handles_staggered_histories(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100, 101, 102, 103, 104, 105],
                "B": [100, 99, 101, 100, 102, 103],
                "C": [100, np.nan, 101, np.nan, 102, np.nan],
                "D": [np.nan, 100, np.nan, 101, np.nan, 102],
            },
            index=pd.date_range("2026-01-01", periods=6, freq="B"),
        )
        cfg = EstimationConfig(
            vol_span=2,
            covariance_window=5,
            covariance_min_periods=3,
            cleaning_method="linear_shrinkage",
            linear_shrinkage=0.1,
            max_abs_return=1.0,
        )

        panel = estimate_clean_covariance_panel(prices, cfg, target_dates=pd.DatetimeIndex([prices.index[-1]]))

        self.assertIn(prices.index[-1], panel)
        cov = next(iter(panel.values()))
        self.assertGreaterEqual(len(cov.index), 2)
        self.assertEqual(list(cov.index), list(cov.columns))
        self.assertIn("A", cov.index)
        self.assertIn("B", cov.index)

    def test_estimate_clean_correlation_panel_returns_valid_correlation(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100, 101, 102, 103, 104, 105],
                "B": [100, 99, 101, 100, 102, 103],
                "C": [100, 100.5, 100.0, 99.8, 100.2, 100.9],
            },
            index=pd.date_range("2026-01-01", periods=6, freq="B"),
        )
        cfg = EstimationConfig(
            vol_span=2,
            covariance_window=5,
            covariance_min_periods=3,
            cleaning_method="linear_shrinkage",
            linear_shrinkage=0.1,
        )

        panel = estimate_clean_correlation_panel(prices, cfg, target_dates=pd.DatetimeIndex([prices.index[-1]]))

        self.assertIn(prices.index[-1], panel)
        corr = panel[prices.index[-1]]
        np.testing.assert_allclose(np.diag(corr), np.ones(len(corr)))
        self.assertTrue(np.allclose(corr.to_numpy(), corr.to_numpy().T))


if __name__ == "__main__":
    unittest.main()
