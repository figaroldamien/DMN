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

from optimal_tf.portfolios import (  # noqa: E402
    agnostic_risk_parity_weights_from_cov,
    naive_markowitz_weights_from_cov,
    risk_parity_weights_from_cov,
)


class PortfolioWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cov = pd.DataFrame(
            [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]],
            index=list("ABC"),
            columns=list("ABC"),
        )

    def test_risk_parity_weights_sum_to_one_in_absolute_value(self) -> None:
        weights = risk_parity_weights_from_cov(self.cov)

        self.assertAlmostEqual(float(weights.abs().sum()), 1.0)
        self.assertGreater(weights["A"], weights["B"])
        self.assertGreater(weights["B"], weights["C"])

    def test_agnostic_risk_parity_weights_are_finite_and_normalized(self) -> None:
        weights = agnostic_risk_parity_weights_from_cov(self.cov)

        self.assertTrue(np.isfinite(weights.to_numpy()).all())
        self.assertAlmostEqual(float(weights.abs().sum()), 1.0)
        self.assertListEqual(list(weights.index), list(self.cov.index))

    def test_naive_markowitz_weights_are_finite_and_normalized(self) -> None:
        weights = naive_markowitz_weights_from_cov(self.cov)

        self.assertTrue(np.isfinite(weights.to_numpy()).all())
        self.assertAlmostEqual(float(weights.abs().sum()), 1.0)
        self.assertListEqual(list(weights.index), list(self.cov.index))


if __name__ == "__main__":
    unittest.main()
