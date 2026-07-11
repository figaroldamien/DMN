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

from optimal_tf.config import BacktestConfig, EstimationConfig, EvaluationConfig, OutputConfig, UniverseConfig  # noqa: E402
from optimal_tf.data_quality import PriceQualityFilterReport  # noqa: E402
from optimal_tf.estimators.cleaning import CorrelationCleanerResult  # noqa: E402
from optimal_tf.estimators.rie_spectral import RieSpectralDecomposition  # noqa: E402
from optimal_tf.services.inspection import _absolute_vector_alignment, _aggregate_matrix_by_groups, _variogram_frame  # noqa: E402
from optimal_tf.services.inspection import run_inspection_interval, run_inspection_snapshot  # noqa: E402
from optimal_tf.services.models import InspectionIntervalRequest, InspectionSnapshotRequest  # noqa: E402


class InspectionTests(unittest.TestCase):
    @staticmethod
    def _quality_report(prices: pd.DataFrame, *, universe: str, start: str, reference_start: str) -> PriceQualityFilterReport:
        tickers = tuple(str(column) for column in prices.columns)
        return PriceQualityFilterReport(
            enabled=True,
            universe=universe,
            start=start,
            reference_start=reference_start,
            reference_rows=int(len(prices.index)),
            total_tickers=int(len(prices.columns)),
            kept_tickers=tickers,
            excluded_tickers=(),
        )

    def test_absolute_vector_alignment_aligns_on_common_tickers(self) -> None:
        current = pd.Series({"A": 0.6, "B": 0.8, "C": 0.0})
        previous = pd.Series({"A": 0.3, "B": 0.4, "D": 0.9})

        alignment = _absolute_vector_alignment(current, previous)

        self.assertIsNotNone(alignment)
        self.assertAlmostEqual(float(alignment), 1.0, places=8)

    def test_absolute_vector_alignment_returns_none_without_common_tickers(self) -> None:
        current = pd.Series({"A": 1.0, "B": 0.0})
        previous = pd.Series({"C": 1.0, "D": 0.0})

        alignment = _absolute_vector_alignment(current, previous)

        self.assertIsNone(alignment)

    def test_variogram_frame_computes_semivariance_by_rank_and_lag(self) -> None:
        frame = pd.DataFrame(
            {
                "rank_1": [1.0, 2.0, 4.0],
                "rank_2": [3.0, 3.0, 3.0],
            },
            index=pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"]),
        )

        variogram = _variogram_frame(frame, max_lag=5)

        self.assertEqual(sorted(variogram["rank"].unique().tolist()), [1, 2])
        rank1_lag1 = variogram.loc[(variogram["rank"] == 1) & (variogram["lag"] == 1), "semivariance"].iloc[0]
        rank1_lag2 = variogram.loc[(variogram["rank"] == 1) & (variogram["lag"] == 2), "semivariance"].iloc[0]
        rank2_lag1 = variogram.loc[(variogram["rank"] == 2) & (variogram["lag"] == 1), "semivariance"].iloc[0]
        self.assertAlmostEqual(float(rank1_lag1), 1.25, places=8)
        self.assertAlmostEqual(float(rank1_lag2), 4.5, places=8)
        self.assertAlmostEqual(float(rank2_lag1), 0.0, places=8)

    def test_aggregate_matrix_by_groups_averages_cross_blocks_and_excludes_same_ticker_diagonal(self) -> None:
        matrix = pd.DataFrame(
            [
                [1.0, 0.2, 0.3],
                [0.2, 1.0, 0.5],
                [0.3, 0.5, 1.0],
            ],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        metadata = pd.DataFrame(
            {
                "sector": ["Tech", "Tech", "Health"],
                "sub_sector": ["Software", "Hardware", "Biotech"],
                "category": ["Tech", "Tech", "Health"],
                "sub_category": ["Software", "Hardware", "Biotech"],
            },
            index=["A", "B", "C"],
        )

        aggregated, pair_counts, membership = _aggregate_matrix_by_groups(matrix, metadata, level="sector")

        self.assertEqual(list(aggregated.index), ["Tech", "Health"])
        self.assertAlmostEqual(float(aggregated.loc["Tech", "Tech"]), 0.2, places=8)
        self.assertAlmostEqual(float(aggregated.loc["Tech", "Health"]), 0.4, places=8)
        self.assertAlmostEqual(float(aggregated.loc["Health", "Health"]), 1.0, places=8)
        self.assertEqual(float(pair_counts.loc["Tech", "Tech"]), 2.0)
        self.assertEqual(float(pair_counts.loc["Tech", "Health"]), 2.0)
        self.assertEqual(float(pair_counts.loc["Health", "Health"]), 1.0)
        self.assertEqual(list(membership.columns), ["group", "sector", "num_tickers", "ticker"])
        self.assertEqual(membership.loc[membership["group"] == "Tech", "num_tickers"].iloc[0], 2)

    def test_run_inspection_interval_skips_dates_without_available_sample(self) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0], "B": [100.0, 99.0]},
            index=pd.DatetimeIndex(["2015-01-30", "2015-02-27"]),
        )
        corr = pd.DataFrame(
            [[1.0, 0.2], [0.2, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        cov = pd.DataFrame(
            [[0.01, 0.002], [0.002, 0.04]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        sample_frame = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, -0.01]},
            index=pd.DatetimeIndex(["2015-02-26", "2015-02-27"]),
        )
        request = InspectionIntervalRequest(
            cleaning_method="empirical",
            estimator_window=60,
            output_dir=None,
        )
        config = (
            UniverseConfig(name="cac40", start="2015-01-01"),
            EstimationConfig(covariance_window=60, covariance_min_periods=2, cleaning_method="empirical"),
            BacktestConfig(),
            object(),
            EvaluationConfig(rebalance_frequency="monthly", evaluation_start="2015-01-01", evaluation_end="2015-02-28"),
            object(),
            OutputConfig(),
        )

        with (
            patch("optimal_tf.services.inspection.load_config", return_value=config),
            patch(
                "optimal_tf.services.inspection.load_filtered_prices_for_universe",
                return_value=(prices, self._quality_report(prices, universe="cac40", start="2015-01-01", reference_start="2015-01-01")),
            ),
            patch("optimal_tf.services.inspection.resolve_target_dates", return_value=list(prices.index)),
            patch(
                "optimal_tf.services.inspection.matrix_sample_bundle",
                side_effect=[
                    ValueError("No correlation sample available on 2015-01-30 for covariance_window=60."),
                    (corr, cov, 2, sample_frame),
                ],
            ),
        ):
            result = run_inspection_interval(request)

        self.assertEqual(list(result.summary_frame["date"]), ["2015-02-27"])
        self.assertEqual(result.observation_dates, (pd.Timestamp("2015-02-27"),))
        self.assertIn("lag", result.variogram_frame.columns)
        self.assertTrue(np.isfinite(result.eigenvector_similarity_frame["abs_alignment_anchor"]).all())

    def test_run_inspection_interval_maps_product_spec_to_engine_inputs(self) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 98.0]},
            index=pd.DatetimeIndex(["2015-01-30", "2015-02-27", "2015-03-31"]),
        )
        corr = pd.DataFrame(
            [[1.0, 0.1], [0.1, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        cov = pd.DataFrame(
            [[0.01, 0.001], [0.001, 0.04]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        sample_frame = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, -0.01]},
            index=pd.DatetimeIndex(["2015-03-30", "2015-03-31"]),
        )
        request = InspectionIntervalRequest(
            cleaning_method="rie_spectral",
            input_type="raw_returns",
            matrix_type="covariance",
            estimator_method="ewma",
            estimator_window=40,
            linear_shrinkage_intensity=0.25,
            output_dir=None,
        )
        config = (
            UniverseConfig(name="cac40", start="2015-01-01"),
            EstimationConfig(covariance_window=60, covariance_min_periods=2, cleaning_method="empirical"),
            BacktestConfig(),
            object(),
            EvaluationConfig(rebalance_frequency="monthly", evaluation_start="2015-01-01", evaluation_end="2015-03-31"),
            object(),
            OutputConfig(),
        )

        with (
            patch("optimal_tf.services.inspection.load_config", return_value=config),
            patch(
                "optimal_tf.services.inspection.load_filtered_prices_for_universe",
                return_value=(prices, self._quality_report(prices, universe="cac40", start="2015-01-01", reference_start="2015-01-01")),
            ),
            patch("optimal_tf.services.inspection.resolve_target_dates", return_value=[prices.index[-1]]),
            patch("optimal_tf.services.inspection.clean_correlation_matrix", return_value=corr) as clean_mock,
            patch("optimal_tf.services.inspection.matrix_sample_bundle", return_value=(corr, cov, 2, sample_frame)) as sample_mock,
        ):
            result = run_inspection_interval(request)

        sample_mock.assert_called_once()
        self.assertEqual(sample_mock.call_args.kwargs["input_type"], "raw")
        self.assertEqual(sample_mock.call_args.kwargs["estimator_method"], "ewma_cross")
        self.assertEqual(sample_mock.call_args.kwargs["estimator_window"], 40)
        clean_mock.assert_called()
        self.assertEqual(clean_mock.call_args.kwargs["method"], "rie_spectral")
        self.assertAlmostEqual(float(clean_mock.call_args.kwargs["linear_shrinkage"]), 0.25, places=8)
        self.assertEqual(result.input_type, "raw_returns")
        self.assertEqual(result.matrix_type, "covariance")
        self.assertEqual(result.estimator_method, "ewma")
        self.assertEqual(result.estimator_window, 40)
        self.assertEqual(result.cleaning_method, "rie_spectral")

    def test_run_inspection_snapshot_uses_rich_cleaner_api_for_correlation_spectrum(self) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 98.0]},
            index=pd.DatetimeIndex(["2015-01-30", "2015-02-27", "2015-03-31"]),
        )
        corr = pd.DataFrame(
            [[1.0, 0.1], [0.1, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        cov = pd.DataFrame(
            [[0.01, 0.001], [0.001, 0.04]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        sample_frame = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.03, -0.01]},
            index=pd.DatetimeIndex(["2015-03-30", "2015-03-31"]),
        )
        request = InspectionSnapshotRequest(
            cleaning_method="rie_spectral",
            estimator_window=40,
            output_dir=None,
        )
        config = (
            UniverseConfig(name="cac40", start="2015-01-01"),
            EstimationConfig(covariance_window=60, covariance_min_periods=2, cleaning_method="empirical"),
            BacktestConfig(),
            object(),
            EvaluationConfig(rebalance_frequency="monthly", evaluation_start="2015-01-01", evaluation_end="2015-03-31"),
            object(),
            OutputConfig(),
        )
        empirical_vectors = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["rank_1", "rank_2"],
        )
        selected_vectors = pd.DataFrame(
            [[-1.0, 0.0], [0.0, 1.0]],
            index=["A", "B"],
            columns=["rank_1", "rank_2"],
        )
        empirical_result = CorrelationCleanerResult(
            method="empirical",
            matrix_type="correlation",
            sample_size=2,
            input_matrix=corr,
            empirical=RieSpectralDecomposition(
                eigenvalues=np.array([1.1, 0.9]),
                eigenvectors=empirical_vectors,
                rank_order=(0, 1),
            ),
            cleaned_eigenvalues_by_input_order=np.array([1.1, 0.9]),
            cleaned=RieSpectralDecomposition(
                eigenvalues=np.array([1.1, 0.9]),
                eigenvectors=empirical_vectors,
                rank_order=(0, 1),
            ),
            cleaned_matrix_pre_projection=corr,
            cleaned_matrix=corr,
            postprocess_applied=False,
            postprocess_steps=(),
            spectral_source="matrix_diagonalization",
        )
        selected_result = CorrelationCleanerResult(
            method="rie_spectral",
            matrix_type="correlation",
            sample_size=2,
            input_matrix=corr,
            empirical=empirical_result.empirical,
            cleaned_eigenvalues_by_input_order=np.array([1.05, 0.95]),
            cleaned=RieSpectralDecomposition(
                eigenvalues=np.array([1.05, 0.95]),
                eigenvectors=selected_vectors,
                rank_order=(0, 1),
            ),
            cleaned_matrix_pre_projection=corr,
            cleaned_matrix=corr,
            postprocess_applied=False,
            postprocess_steps=(),
            spectral_source="empirical_vectors_plus_xi_hat",
        )

        with (
            patch("optimal_tf.services.inspection.load_config", return_value=config),
            patch(
                "optimal_tf.services.inspection.load_filtered_prices_for_universe",
                return_value=(prices, self._quality_report(prices, universe="cac40", start="2015-01-01", reference_start="2015-01-01")),
            ),
            patch("optimal_tf.services.inspection.matrix_sample_bundle", return_value=(corr, cov, 2, sample_frame)),
            patch("optimal_tf.services.inspection.clean_correlation_matrix_rich", side_effect=[empirical_result, selected_result]) as clean_mock,
        ):
            result = run_inspection_snapshot(request)

        self.assertEqual(clean_mock.call_count, 2)
        np.testing.assert_allclose(
            result.correlation_eigenvectors["corr_ev1"].to_numpy(dtype=float),
            np.array([-1.0, 0.0]),
        )
        np.testing.assert_allclose(
            result.correlation_spectrum["eigenvalue"].to_numpy(dtype=float)[:2],
            np.array([1.05, 0.95]),
        )


if __name__ == "__main__":
    unittest.main()
