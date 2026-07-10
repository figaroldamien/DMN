from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.dashboard_workspace import (  # noqa: E402
    build_workspace_context,
    normalize_workspace_selection,
    workspace_defaults_from_config,
    workspace_mode_uses_shared_controls,
)


class DashboardWorkspaceTests(unittest.TestCase):
    def test_workspace_defaults_follow_config(self) -> None:
        defaults = workspace_defaults_from_config(
            {
                "universe": {"name": "sp500", "start": "2010-01-01"},
                "evaluation": {"evaluation_start": "2020-01-01", "evaluation_end": "2024-12-31"},
            },
            default_config_path="configs/optimal_tf.example.toml",
            fallback_universe="cac40",
        )

        self.assertEqual(defaults.config_path, "configs/optimal_tf.example.toml")
        self.assertEqual(defaults.universe, "sp500")
        self.assertEqual(defaults.start, "2010-01-01")
        self.assertEqual(defaults.evaluation_start, "2020-01-01")
        self.assertEqual(defaults.evaluation_end, "2024-12-31")

    def test_workspace_selection_keeps_valid_stored_values(self) -> None:
        group, options, universe = normalize_workspace_selection(
            universe_groups={"Markets": ["cac40", "sp500"], "Index": ["world_index"]},
            fallback_universe_options=["cac40", "sp500", "world_index"],
            universe_default="sp500",
            stored_group="Markets",
            stored_universe="sp500",
        )

        self.assertEqual(group, "Markets")
        self.assertEqual(options, ["cac40", "sp500"])
        self.assertEqual(universe, "sp500")

    def test_workspace_selection_falls_back_when_stored_values_are_invalid(self) -> None:
        group, options, universe = normalize_workspace_selection(
            universe_groups={"Markets": ["cac40", "sp500"], "Index": ["world_index"]},
            fallback_universe_options=["cac40", "sp500", "world_index"],
            universe_default="world_index",
            stored_group="Unknown",
            stored_universe="nasdaq100",
        )

        self.assertEqual(group, "Index")
        self.assertEqual(options, ["world_index"])
        self.assertEqual(universe, "world_index")

    def test_workspace_mode_controls_are_hidden_for_workspace_and_guide(self) -> None:
        self.assertFalse(workspace_mode_uses_shared_controls("Workspace"))
        self.assertFalse(workspace_mode_uses_shared_controls("Guide"))
        self.assertTrue(workspace_mode_uses_shared_controls("Run"))
        self.assertTrue(workspace_mode_uses_shared_controls("Matrix Inspection"))
        self.assertTrue(workspace_mode_uses_shared_controls("Compare"))

    def test_build_workspace_context_normalizes_optional_dates(self) -> None:
        context = build_workspace_context(
            config_path="configs/optimal_tf.example.toml",
            config_defaults={"universe": {"name": "cac40"}},
            universe_group="Markets",
            universe="cac40",
            start="2015-01-01",
            evaluation_start="",
            evaluation_end=None,
            refresh_pending=True,
        )

        self.assertEqual(context.universe_group, "Markets")
        self.assertEqual(context.universe, "cac40")
        self.assertEqual(context.start, "2015-01-01")
        self.assertIsNone(context.evaluation_start)
        self.assertIsNone(context.evaluation_end)
        self.assertTrue(context.refresh_pending)


if __name__ == "__main__":
    unittest.main()
