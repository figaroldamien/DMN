from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.config import AllocationConfig, BacktestConfig, EstimationConfig, EvaluationConfig, OutputConfig, UniverseConfig  # noqa: E402
from optimal_tf.services.models import StrategyTestbedRequest  # noqa: E402
from optimal_tf.services.standard import run_strategy_testbed  # noqa: E402
from optimal_tf.strategies.arp import agnostic_risk_parity_weights_from_cov  # noqa: E402
from optimal_tf.strategies_agnostic.api import (  # noqa: E402
    agnostic_recipe_state_at_date,
    agnostic_strategy_state_at_date,
    compute_agnostic_recipe_panel,
)
from optimal_tf.strategies_agnostic.catalog import supported_agnostic_strategies  # noqa: E402
from optimal_tf.strategies_agnostic.normalization import normalize_by_gross_exposure  # noqa: E402
from optimal_tf.strategies_agnostic.position_engine import build_agnostic_positions  # noqa: E402
from optimal_tf.strategies_agnostic.q_models import (  # noqa: E402
    clean_q_matrix,
    clean_structural_matrix,
    correlation_q_matrix,
    empirical_signal_q_matrix,
    identity_q_matrix,
    phi_shrink_correlation_q_matrix,
    q_matrix_kind,
    resolve_q_matrix,
)
from optimal_tf.strategies_agnostic.signals import resolve_signal, resolve_signal_panel  # noqa: E402


class AgnosticPositionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corr = pd.DataFrame(
            [[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]],
            index=list("ABC"),
            columns=list("ABC"),
        )
        self.signal = pd.Series(1.0, index=self.corr.index, dtype=float)

    def test_identity_q_reproduces_current_arp_shape_after_gross_normalization(self) -> None:
        raw = build_agnostic_positions(self.corr, identity_q_matrix(self.corr), self.signal)
        weights = normalize_by_gross_exposure(raw)
        arp = agnostic_risk_parity_weights_from_cov(self.corr)
        self.assertTrue(np.allclose(weights.to_numpy(), arp.to_numpy()))

    def test_correlation_q_matches_inverse_correlation_engine(self) -> None:
        raw = build_agnostic_positions(self.corr, correlation_q_matrix(self.corr), self.signal)
        expected = np.linalg.pinv(self.corr.to_numpy(dtype=float)) @ np.ones(len(self.corr), dtype=float)
        self.assertTrue(np.allclose(raw.to_numpy(), expected, atol=1e-8))

    def test_phi_shrink_q_interpolates_between_identity_and_correlation(self) -> None:
        q0 = phi_shrink_correlation_q_matrix(self.corr, phi=0.0)
        q1 = phi_shrink_correlation_q_matrix(self.corr, phi=1.0)
        self.assertTrue(np.allclose(q0.to_numpy(), identity_q_matrix(self.corr).to_numpy()))
        self.assertTrue(np.allclose(q1.to_numpy(), self.corr.to_numpy()))

    def test_clean_structural_matrix_returns_valid_correlation(self) -> None:
        skewed = pd.DataFrame(
            [[1.0, 0.8, 0.3], [0.8, 1.2, 0.4], [0.3, 0.4, 0.9]],
            index=list("ABC"),
            columns=list("ABC"),
        )
        cleaned = clean_structural_matrix(skewed)
        np.testing.assert_allclose(np.diag(cleaned), np.ones(3))
        self.assertTrue(np.allclose(cleaned.to_numpy(), cleaned.to_numpy().T))

    def test_clean_q_matrix_dispatches_structural_case(self) -> None:
        q = phi_shrink_correlation_q_matrix(self.corr, phi=0.5)
        cleaned = clean_q_matrix(q, q_kind=q_matrix_kind("phi_shrink_correlation"), est_cfg=EstimationConfig())
        np.testing.assert_allclose(np.diag(cleaned), np.ones(3))
        self.assertTrue(np.allclose(cleaned.to_numpy(), cleaned.to_numpy().T))

    def test_resolve_q_matrix_builds_and_cleans_through_one_interface(self) -> None:
        cleaned = resolve_q_matrix(
            self.corr,
            q_model="phi_shrink_correlation",
            phi=0.5,
            est_cfg=EstimationConfig(),
        )
        np.testing.assert_allclose(np.diag(cleaned), np.ones(3))
        self.assertTrue(np.allclose(cleaned.to_numpy(), cleaned.to_numpy().T))

    def test_empirical_signal_q_matrix_uses_signal_history(self) -> None:
        signal_panel = pd.DataFrame(
            {
                "A": [0.1, 0.2, 0.15, -0.05],
                "B": [0.0, 0.1, 0.05, -0.02],
                "C": [-0.2, -0.1, -0.05, 0.01],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="B"),
        )
        q = empirical_signal_q_matrix(signal_panel, assets=list("ABC"))
        np.testing.assert_allclose(np.diag(q), np.ones(3))
        self.assertEqual(list(q.index), list("ABC"))

    def test_resolve_q_matrix_supports_empirical_model(self) -> None:
        signal_panel = pd.DataFrame(
            {
                "A": [0.1, 0.2, 0.15, -0.05],
                "B": [0.0, 0.1, 0.05, -0.02],
                "C": [-0.2, -0.1, -0.05, 0.01],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="B"),
        )
        cleaned = resolve_q_matrix(
            self.corr,
            q_model="empirical",
            phi=0.0,
            est_cfg=EstimationConfig(cleaning_method="linear_shrinkage", linear_shrinkage=0.1),
            signal_panel=signal_panel,
            sample_size=len(signal_panel),
        )
        np.testing.assert_allclose(np.diag(cleaned), np.ones(3))
        self.assertTrue(np.allclose(cleaned.to_numpy(), cleaned.to_numpy().T))

    def test_resolve_q_matrix_supports_empirical_model_with_rie_reference(self) -> None:
        signal_panel = pd.DataFrame(
            {
                "A": [0.1, 0.2, 0.15, -0.05],
                "B": [0.0, 0.1, 0.05, -0.02],
                "C": [-0.2, -0.1, -0.05, 0.01],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="B"),
        )
        try:
            cleaned = resolve_q_matrix(
                self.corr,
                q_model="empirical",
                phi=0.0,
                est_cfg=EstimationConfig(cleaning_method="rie_reference"),
                signal_panel=signal_panel,
                sample_size=len(signal_panel),
            )
        except ModuleNotFoundError:
            self.skipTest("rie_estimator optional package is not installed in this environment.")
        np.testing.assert_allclose(np.diag(cleaned), np.ones(3))
        self.assertTrue(np.allclose(cleaned.to_numpy(), cleaned.to_numpy().T))


class AgnosticStrategyApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0, 104.5, 105.0],
                "B": [100.0, 100.5, 99.5, 100.0, 100.8, 101.1],
                "C": [100.0, 99.7, 100.2, 100.9, 101.2, 101.8],
            },
            index=pd.date_range("2026-01-01", periods=6, freq="B"),
        )
        self.cov = pd.DataFrame(
            [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]],
            index=list("ABC"),
            columns=list("ABC"),
        )

    def test_state_uses_existing_covariance_cache_and_keeps_raw_amplitude(self) -> None:
        state = agnostic_strategy_state_at_date(
            self.prices,
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2),
            date=self.prices.index[-1],
            covariance_cache={self.prices.index[-1]: self.cov},
            signal_model="ones",
            q_model="identity",
            omega=2.5,
            normalization="raw",
        )
        self.assertTrue(np.isfinite(state.base_weights.to_numpy()).all())
        self.assertAlmostEqual(float(state.signal_scale), 2.5)
        self.assertTrue(np.allclose(state.base_weights.to_numpy(), state.effective_weights.to_numpy()))

    def test_resolve_signal_builds_ones_vector_on_available_assets(self) -> None:
        signal = resolve_signal(
            self.prices,
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2),
            date=self.prices.index[-1],
            corr=pd.DataFrame(np.eye(3), index=list("ABC"), columns=list("ABC")),
            signal_model="ones",
        )
        self.assertListEqual(list(signal.index), list("ABC"))
        self.assertTrue((signal == 1.0).all())

    def test_resolve_signal_panel_builds_trend_history(self) -> None:
        panel = resolve_signal_panel(
            self.prices,
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2),
            date=self.prices.index[-1],
            corr=pd.DataFrame(np.eye(3), index=list("ABC"), columns=list("ABC")),
            signal_model="trend_ema",
        )
        self.assertListEqual(list(panel.columns), list("ABC"))
        self.assertEqual(panel.index[-1], self.prices.index[-1])

    def test_trend_signal_path_produces_finite_gross_normalized_weights(self) -> None:
        state = agnostic_strategy_state_at_date(
            self.prices,
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2),
            date=self.prices.index[-1],
            covariance_cache={self.prices.index[-1]: self.cov},
            signal_model="trend_ema",
            q_model="phi_shrink_correlation",
            phi=0.3,
            normalization="gross",
        )
        self.assertTrue(np.isfinite(state.effective_weights.to_numpy()).all())
        self.assertAlmostEqual(float(state.effective_weights.abs().sum()), 1.0)

    def test_rie_reference_config_still_produces_weights_with_structural_q(self) -> None:
        state = agnostic_strategy_state_at_date(
            self.prices,
            EstimationConfig(
                vol_span=2,
                trend_span=2,
                covariance_min_periods=2,
                cleaning_method="rie_reference",
            ),
            date=self.prices.index[-1],
            covariance_cache={self.prices.index[-1]: self.cov},
            signal_model="ones",
            q_model="phi_shrink_correlation",
            phi=0.5,
            normalization="gross",
        )
        self.assertTrue(np.isfinite(state.effective_weights.to_numpy()).all())
        self.assertGreater(float(state.effective_weights.abs().sum()), 0.0)

    def test_supported_recipe_names_cover_core_lab_cases(self) -> None:
        names = supported_agnostic_strategies()
        self.assertIn("ARP_AGNOSTIC", names)
        self.assertIn("MARKOWITZ_AGNOSTIC", names)
        self.assertIn("ATF_AGNOSTIC", names)
        self.assertIn("ATF_EMPIRICAL_Q", names)
        self.assertIn("PHI_25", names)
        self.assertIn("PHI_50", names)

    def test_named_recipe_matches_direct_agnostic_builder(self) -> None:
        direct = agnostic_strategy_state_at_date(
            self.prices,
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2),
            date=self.prices.index[-1],
            covariance_cache={self.prices.index[-1]: self.cov},
            signal_model="ones",
            q_model="identity",
            normalization="gross",
        )
        named = agnostic_recipe_state_at_date(
            self.prices,
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2),
            "ARP_AGNOSTIC",
            date=self.prices.index[-1],
            covariance_cache={self.prices.index[-1]: self.cov},
        )
        self.assertTrue(np.allclose(direct.effective_weights.to_numpy(), named.effective_weights.to_numpy()))

    def test_recipe_panel_returns_strategy_panel_shape(self) -> None:
        idx = self.prices.index[-2:]
        panel = compute_agnostic_recipe_panel(
            self.prices,
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2),
            "PHI_50",
            target_dates=idx,
            covariance_cache={self.prices.index[-1]: self.cov},
        )
        self.assertListEqual(list(panel.base_weights.index), list(idx))
        self.assertListEqual(list(panel.effective_weights.columns), list(self.prices.columns))
        self.assertTrue(np.isfinite(panel.signal_scale.to_numpy()).all())


class StrategyTestbedServiceTests(unittest.TestCase):
    def test_run_strategy_testbed_uses_named_strategy_panel_when_strategy_is_provided(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0],
                "B": [100.0, 99.5, 100.5, 101.0],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="B"),
        )
        config_tuple = (
            UniverseConfig(name="test", start="2026-01-01"),
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2, cleaning_method="empirical"),
            BacktestConfig(portfolio_vol_target=False, long_only=False),
            AllocationConfig(strategy="RP"),
            EvaluationConfig(strategy="RP", rebalance_frequency="monthly"),
            SimpleNamespace(),
            OutputConfig(evaluation_dir=None, evaluation_plot=False),
        )
        benchmark = pd.Series([0.0, 0.01], index=prices.index[-2:], dtype=float)
        captured: dict[str, object] = {}

        def fake_engine(prices_frame, est_cfg, bt_cfg, eval_cfg, *, compute_strategy_panel_fn, estimate_clean_covariance_panel_fn):
            del estimate_clean_covariance_panel_fn
            captured["strategy_label"] = eval_cfg.strategy
            compute_strategy_panel_fn(
                prices_frame,
                est_cfg,
                eval_cfg.strategy,
                long_only=bt_cfg.long_only,
                target_dates=prices_frame.index[-2:],
                covariance_cache={prices_frame.index[-1]: pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])},
            )
            return SimpleNamespace(daily_returns_net=pd.Series([0.01, -0.005], index=prices_frame.index[-2:], dtype=float))

        with patch("optimal_tf.services.standard.load_config", return_value=config_tuple):
            with patch("optimal_tf.services.standard.load_prices_for_universe", return_value=prices):
                with patch("optimal_tf.services.standard.compute_strategy_panel") as mocked_panel:
                    mocked_panel.return_value = SimpleNamespace(
                        base_weights=pd.DataFrame(0.0, index=prices.index[-2:], columns=prices.columns),
                        effective_weights=pd.DataFrame(0.0, index=prices.index[-2:], columns=prices.columns),
                        signal_scale=pd.Series(1.0, index=prices.index[-2:], dtype=float),
                    )
                    with patch("optimal_tf.services.standard._engine_evaluate_portfolio", side_effect=fake_engine):
                        with patch("optimal_tf.services.standard._load_primary_benchmark_returns", return_value=(benchmark, "bench", None)):
                            with patch("optimal_tf.services.standard.equal_weight_buy_and_hold_benchmark", return_value=benchmark):
                                result = run_strategy_testbed(
                                    StrategyTestbedRequest(
                                        config_path="configs/optimal_tf.example.toml",
                                        universe="test",
                                        strategy="RP",
                                        output_dir=None,
                                        output_plot=False,
                                    )
                                )

        mocked_panel.assert_called_once()
        self.assertEqual(mocked_panel.call_args.args[2], "RP")
        self.assertEqual(str(captured["strategy_label"]), "RP")
        self.assertEqual(result.strategy_label, "RP")

    def test_run_strategy_testbed_injects_custom_agnostic_panel(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0],
                "B": [100.0, 99.5, 100.5, 101.0],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="B"),
        )
        config_tuple = (
            UniverseConfig(name="test", start="2026-01-01"),
            EstimationConfig(vol_span=2, trend_span=2, covariance_min_periods=2, cleaning_method="empirical"),
            BacktestConfig(portfolio_vol_target=False, long_only=False),
            AllocationConfig(strategy="RP"),
            EvaluationConfig(strategy="RP", rebalance_frequency="monthly"),
            SimpleNamespace(),
            OutputConfig(evaluation_dir=None, evaluation_plot=False),
        )
        benchmark = pd.Series([0.0, 0.01], index=prices.index[-2:], dtype=float)
        captured: dict[str, object] = {}

        def fake_engine(prices_frame, est_cfg, bt_cfg, eval_cfg, *, compute_strategy_panel_fn, estimate_clean_covariance_panel_fn):
            del estimate_clean_covariance_panel_fn
            captured["strategy_label"] = eval_cfg.strategy
            compute_strategy_panel_fn(
                prices_frame,
                est_cfg,
                eval_cfg.strategy,
                long_only=bt_cfg.long_only,
                target_dates=prices_frame.index[-2:],
                covariance_cache={prices_frame.index[-1]: pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"])},
            )
            return SimpleNamespace(daily_returns_net=pd.Series([0.01, -0.005], index=prices_frame.index[-2:], dtype=float))

        with patch("optimal_tf.services.standard.load_config", return_value=config_tuple):
            with patch("optimal_tf.services.standard.load_prices_for_universe", return_value=prices):
                with patch("optimal_tf.services.standard.compute_agnostic_panel") as mocked_panel:
                    mocked_panel.return_value = SimpleNamespace(
                        base_weights=pd.DataFrame(0.0, index=prices.index[-2:], columns=prices.columns),
                        effective_weights=pd.DataFrame(0.0, index=prices.index[-2:], columns=prices.columns),
                        signal_scale=pd.Series(1.0, index=prices.index[-2:], dtype=float),
                    )
                    with patch("optimal_tf.services.standard._engine_evaluate_portfolio", side_effect=fake_engine):
                        with patch("optimal_tf.services.standard._load_primary_benchmark_returns", return_value=(benchmark, "bench", None)):
                            with patch("optimal_tf.services.standard.equal_weight_buy_and_hold_benchmark", return_value=benchmark):
                                result = run_strategy_testbed(
                                    StrategyTestbedRequest(
                                        config_path="configs/optimal_tf.example.toml",
                                        universe="test",
                                        signal_model="trend_ema",
                                        q_model="phi_shrink_correlation",
                                        phi=0.35,
                                        omega=1.7,
                                        normalization="gross",
                                        output_dir=None,
                                        output_plot=False,
                                    )
                                )

        mocked_panel.assert_called_once()
        self.assertEqual(mocked_panel.call_args.kwargs["signal_model"], "trend_ema")
        self.assertEqual(mocked_panel.call_args.kwargs["q_model"], "phi_shrink_correlation")
        self.assertAlmostEqual(float(mocked_panel.call_args.kwargs["phi"]), 0.35)
        self.assertAlmostEqual(float(mocked_panel.call_args.kwargs["omega"]), 1.7)
        self.assertEqual(mocked_panel.call_args.kwargs["normalization"], "gross")
        self.assertIn("phi=0.35", str(captured["strategy_label"]))
        self.assertEqual(result.signal_model, "trend_ema")
        self.assertEqual(result.q_model, "phi_shrink_correlation")
        self.assertAlmostEqual(float(result.omega), 1.7)

    def test_run_strategy_testbed_rejects_phi_without_phi_q_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "phi is only supported"):
            run_strategy_testbed(
                StrategyTestbedRequest(
                    q_model="identity",
                    phi=0.2,
                )
            )

    def test_run_strategy_testbed_derives_alpha_from_changed_span(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0],
                "B": [100.0, 99.5, 100.5, 101.0],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="B"),
        )
        base_estimation = EstimationConfig(
            vol_span=2,
            trend_span=252,
            trend_alpha=0.01575,
            covariance_min_periods=2,
            cleaning_method="empirical",
        )
        config_tuple = (
            UniverseConfig(name="test", start="2026-01-01"),
            base_estimation,
            BacktestConfig(portfolio_vol_target=False, long_only=False),
            AllocationConfig(strategy="RP"),
            EvaluationConfig(strategy="RP", rebalance_frequency="monthly"),
            SimpleNamespace(),
            OutputConfig(evaluation_dir=None, evaluation_plot=False),
        )
        benchmark = pd.Series([0.0, 0.01], index=prices.index[-2:], dtype=float)
        captured: dict[str, object] = {}

        def fake_engine(prices_frame, est_cfg, bt_cfg, eval_cfg, *, compute_strategy_panel_fn, estimate_clean_covariance_panel_fn):
            del prices_frame, bt_cfg, eval_cfg, estimate_clean_covariance_panel_fn
            captured["trend_alpha"] = est_cfg.trend_alpha
            captured["trend_span"] = est_cfg.trend_span
            return SimpleNamespace(daily_returns_net=pd.Series([0.01, -0.005], index=prices.index[-2:], dtype=float))

        with patch("optimal_tf.services.standard.load_config", return_value=config_tuple):
            with patch("optimal_tf.services.standard.load_prices_for_universe", return_value=prices):
                with patch("optimal_tf.services.standard._engine_evaluate_portfolio", side_effect=fake_engine):
                    with patch("optimal_tf.services.standard._load_primary_benchmark_returns", return_value=(benchmark, "bench", None)):
                        with patch("optimal_tf.services.standard.equal_weight_buy_and_hold_benchmark", return_value=benchmark):
                            run_strategy_testbed(
                                StrategyTestbedRequest(
                                    config_path="configs/optimal_tf.example.toml",
                                    universe="test",
                                    signal_model="trend_ema",
                                    trend_span=60,
                                    trend_alpha=0.01575,
                                    output_dir=None,
                                    output_plot=False,
                                )
                            )

        self.assertAlmostEqual(float(captured["trend_alpha"]), 2.0 / 61.0)
        self.assertEqual(int(captured["trend_span"]), 60)


if __name__ == "__main__":
    unittest.main()
