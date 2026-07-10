from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.dashboard_catalog import (  # noqa: E402
    COMMON_EVALUATION_DATE_SERVICES,
    MODE_INTRO,
    MODE_SERVICES,
    PRODUCT_MODES,
    SERVICE_INTRO,
    all_service_routes,
    compare_service_names,
    mode_for_service,
)


class DashboardCatalogTests(unittest.TestCase):
    def test_product_modes_follow_target_information_architecture(self) -> None:
        self.assertEqual(
            PRODUCT_MODES,
            ("Workspace", "Run", "Matrix Inspection", "Compare", "Search", "Guide"),
        )

    def test_every_service_route_has_intro(self) -> None:
        self.assertEqual(set(all_service_routes()), set(SERVICE_INTRO))

    def test_every_mode_has_intro(self) -> None:
        self.assertEqual(set(PRODUCT_MODES), set(MODE_INTRO))

    def test_every_service_belongs_to_one_mode(self) -> None:
        seen_services: set[str] = set()
        for mode, services in MODE_SERVICES.items():
            for service_name in services:
                self.assertNotIn(service_name, seen_services)
                seen_services.add(service_name)
                self.assertEqual(mode_for_service(service_name), mode)

    def test_common_evaluation_date_routes_are_valid(self) -> None:
        self.assertTrue(COMMON_EVALUATION_DATE_SERVICES)
        self.assertTrue(COMMON_EVALUATION_DATE_SERVICES.issubset(set(all_service_routes())))

    def test_compare_mode_contains_full_comparison_family(self) -> None:
        self.assertEqual(
            compare_service_names(),
            [
                "Compare",
                "Vary strategy",
                "Vary cleaning",
                "Vary window",
                "Vary frequency",
            ],
        )

    def test_search_mode_keeps_exploration_services_together(self) -> None:
        self.assertEqual(
            list(MODE_SERVICES["Search"]),
            [
                "Strategy testbed",
                "Hyperparameter tuning",
            ],
        )

    def test_matrix_inspection_mode_groups_static_and_interval_views(self) -> None:
        self.assertEqual(
            list(MODE_SERVICES["Matrix Inspection"]),
            [
                "Inspect at date",
                "Inspect over interval",
            ],
        )


if __name__ == "__main__":
    unittest.main()
