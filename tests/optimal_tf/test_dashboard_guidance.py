from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.dashboard_guidance import (  # noqa: E402
    guide_next_step_rows,
    guide_service_choices,
    workspace_overview_rows,
)


class DashboardGuidanceTests(unittest.TestCase):
    def test_workspace_overview_rows_summarize_key_defaults(self) -> None:
        rows = workspace_overview_rows(
            "configs/optimal_tf.example.toml",
            {
                "universe": {"name": "sp500", "start": "2010-01-01"},
                "allocation": {"strategy": "RP"},
                "evaluation": {
                    "strategy": "ARP",
                    "evaluation_start": "2020-01-01",
                    "evaluation_end": "2024-12-31",
                },
                "compare": {"strategies": ["RP", "ARP", "NM"]},
            },
        )
        row_map = {row["field"]: row["value"] for row in rows}

        self.assertEqual(row_map["config_path"], "configs/optimal_tf.example.toml")
        self.assertEqual(row_map["universe"], "sp500")
        self.assertEqual(row_map["evaluation_window"], "2020-01-01 -> 2024-12-31")
        self.assertEqual(row_map["compare_scope"], "3 strategies")

    def test_guide_service_choices_cover_run_compare_and_search(self) -> None:
        choices = guide_service_choices()
        self.assertEqual([row["service_family"] for row in choices], ["Run", "Matrix Inspection", "Compare", "Search"])
        self.assertTrue(all(row["when_to_use"] for row in choices))
        self.assertTrue(all(row["best_for"] for row in choices))

    def test_guide_next_step_rows_cover_main_user_goals(self) -> None:
        rows = guide_next_step_rows()
        services = [row["recommended_service"] for row in rows]
        self.assertIn("Run / Allocation", services)
        self.assertIn("Run / Evaluation", services)
        self.assertIn("Matrix Inspection / Inspect at date", services)
        self.assertIn("Matrix Inspection / Inspect over interval", services)
        self.assertIn("Compare / Comparison Lab", services)
        self.assertIn("Search / Strategy testbed", services)
        self.assertIn("Search / Hyperparameter tuning", services)


if __name__ == "__main__":
    unittest.main()
