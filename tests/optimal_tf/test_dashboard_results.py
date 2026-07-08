from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.dashboard_results import (  # noqa: E402
    build_dashboard_run_summary,
    dashboard_run_summary_rows,
)
from optimal_tf.services import (  # noqa: E402
    CompareRequest,
    RunArtifacts,
    StandardEvaluationRequest,
    StrategyTestbedRequest,
)


class DashboardResultsTests(unittest.TestCase):
    def test_standard_run_summary_uses_request_and_resolved_context(self) -> None:
        summary = build_dashboard_run_summary(
            mode="Run",
            service="Evaluation",
            request=StandardEvaluationRequest(
                universe="sp500",
                strategy="ARP",
                cleaning_method="linear_shrinkage",
                covariance_window=150,
                rebalance_frequency="monthly",
                weight_smoothing_alpha=0.8,
                evaluation_start="2020-01-01",
                evaluation_end="2024-12-31",
                output_dir="output/optimal_tf/dashboard/evaluation",
            ),
            universe="sp500",
            artifacts=RunArtifacts(
                root_dir=Path("output/optimal_tf/dashboard/evaluation/run_001"),
                files={"summary": Path("summary.json"), "nav": Path("nav.csv")},
            ),
            resolved={"strategy": "ARP"},
            highlights={"best_metric": "sharpe"},
            warnings=["benchmark fallback"],
        )

        self.assertEqual(summary.mode, "Run")
        self.assertEqual(summary.service, "Evaluation")
        self.assertEqual(summary.request_type, "StandardEvaluationRequest")
        self.assertEqual(summary.primary_subject, "ARP")
        self.assertEqual(summary.strategy_count, 1)
        self.assertEqual(summary.strategies, ("ARP",))
        self.assertEqual(summary.cleaning_method, "linear_shrinkage")
        self.assertEqual(summary.covariance_window, 150)
        self.assertEqual(summary.rebalance_frequency, "monthly")
        self.assertEqual(summary.weight_smoothing_alpha, 0.8)
        self.assertEqual(summary.artifact_root, "output/optimal_tf/dashboard/evaluation/run_001")
        self.assertEqual(summary.artifact_count, 2)
        self.assertEqual(summary.highlights_count, 1)
        self.assertEqual(summary.warning_count, 1)

    def test_compare_run_summary_tracks_strategy_collections(self) -> None:
        summary = build_dashboard_run_summary(
            mode="Compare",
            service="Compare",
            request=CompareRequest(
                universe="cac40",
                strategies=["RP", "ARP"],
                cleaning_method="rie_reference",
                covariance_window=60,
                rebalance_frequency="weekly",
                weight_smoothing_alpha=1.0,
            ),
            universe="cac40",
            artifacts=RunArtifacts(files={"manifest": Path("manifest.json")}),
            resolved={"strategies": ["RP", "ARP"]},
        )

        self.assertEqual(summary.primary_subject, "2 strategies")
        self.assertEqual(summary.strategy_count, 2)
        self.assertEqual(summary.strategies, ("RP", "ARP"))
        self.assertEqual(summary.artifact_names, ("manifest",))

    def test_testbed_summary_prefers_strategy_label(self) -> None:
        summary = build_dashboard_run_summary(
            mode="Search",
            service="Strategy testbed",
            request=StrategyTestbedRequest(
                universe="nasdaq100",
                strategy=None,
                signal_model="trend_ema",
                q_model="identity",
                cleaning_method="linear_shrinkage",
            ),
            universe="nasdaq100",
            resolved={"strategy_label": "Custom agnostic", "signal_model": "trend_ema"},
        )

        self.assertEqual(summary.primary_subject, "Custom agnostic")
        self.assertEqual(summary.strategy_count, 0)
        self.assertEqual(summary.strategies, ())

    def test_summary_rows_skip_empty_values_and_format_sequences(self) -> None:
        summary = build_dashboard_run_summary(
            mode="Search",
            service="Hyperparameter tuning",
            request={"strategies": ["RP", "ARP"], "methods": ["rie_reference"], "output_dir": ""},
            universe="eurostoxx50",
            artifacts={"files": {"summary": Path("summary.json"), "results": Path("results.csv")}},
        )

        rows = dashboard_run_summary_rows(summary)
        row_map = {row["field"]: row["value"] for row in rows}

        self.assertEqual(row_map["strategies"], "RP, ARP")
        self.assertEqual(row_map["artifact_names"], "results, summary")
        self.assertNotIn("output_dir", row_map)


if __name__ == "__main__":
    unittest.main()
