from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_core.backtest import evaluate_portfolio  # noqa: E402
from trading_core.reporting import EvaluationSummary, cumulative_nav, write_evaluation_outputs  # noqa: E402


class CoreBacktestReportingTests(unittest.TestCase):
    def test_evaluate_portfolio_builds_non_empty_result(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0],
                "B": [100.0, 100.0, 100.0, 100.0],
            },
            index=pd.to_datetime(["2026-01-30", "2026-02-02", "2026-02-27", "2026-03-02"]),
        )

        class EstCfg:
            max_abs_return = 1.0

        class BtCfg:
            sigma_target_annual = 0.15
            portfolio_vol_target = False
            portfolio_vol_span = 60
            cost_bps = 10.0
            long_only = False

        class EvalCfg:
            strategy = "RP"
            rebalance_frequency = "monthly"
            evaluation_start = None
            evaluation_end = None

        weights_panel = pd.DataFrame(
            {
                "A": [0.0, 1.0, 0.0, 0.0],
                "B": [0.0, 0.0, 1.0, 1.0],
            },
            index=prices.index,
        )

        panel = type(
            "Panel",
            (),
            {
                "base_weights": weights_panel,
                "effective_weights": weights_panel,
                "signal_scale": pd.Series(1.0, index=weights_panel.index),
            },
        )()

        result = evaluate_portfolio(
            prices,
            EstCfg(),
            BtCfg(),
            EvalCfg(),
            compute_strategy_panel_fn=lambda *args, **kwargs: panel,
            estimate_clean_covariance_panel_fn=lambda *args, **kwargs: {prices.index[0]: pd.DataFrame()},
        )

        self.assertEqual(int(result.summary.num_rebalances), 2)
        self.assertFalse(result.daily_returns_net.empty)
        self.assertGreater(float(result.costs_by_rebalance.sum()), 0.0)

    def test_reporting_exports_expected_files(self) -> None:
        index = pd.to_datetime(["2026-01-30"])
        result = type(
            "Result",
            (),
            {
                "summary": EvaluationSummary(
                    total_return=0.1,
                    ann_return=0.12,
                    cagr=0.11,
                    ann_vol=0.15,
                    sharpe=0.8,
                    sortino=1.0,
                    skewness=0.1,
                    mar=2.2,
                    mdd=-0.05,
                    avg_turnover=0.2,
                    annualized_turnover=10.0,
                    total_cost=0.001,
                    annualized_cost=0.01,
                    pct_positive_days=0.55,
                    num_days=3,
                    num_rebalances=1,
                    avg_turnover_per_rebalance=1.0,
                    avg_cost_per_rebalance=0.001,
                    total_return_gross=0.101,
                    total_return_cost_drag=0.001,
                ),
                "weights_by_rebalance": pd.DataFrame({"A": [0.6], "B": [0.4]}, index=index),
                "base_weights_by_rebalance": pd.DataFrame({"A": [0.6], "B": [0.4]}, index=index),
                "effective_weights_by_rebalance": pd.DataFrame({"A": [0.6], "B": [0.4]}, index=index),
                "signal_scale_by_rebalance": pd.Series([1.0], index=index),
                "portfolio_vol_scale": pd.Series([1.0], index=index),
                "daily_returns_gross": pd.Series([0.01], index=index),
                "daily_returns_net": pd.Series([0.009], index=index),
                "turnover_by_rebalance": pd.Series([1.0], index=index),
                "costs_by_rebalance": pd.Series([0.001], index=index),
            },
        )()

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            write_evaluation_outputs(result, str(outdir))

            self.assertTrue((outdir / "weights_by_rebalance.csv").exists())
            self.assertTrue((outdir / "summary.json").exists())
            summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total_return"], 0.1)
            self.assertEqual(summary["total_return_gross"], 0.101)
            self.assertEqual(summary["mar"], 2.2)

    def test_cumulative_nav_compounds_returns(self) -> None:
        nav = cumulative_nav(pd.Series([0.10, -0.05], index=pd.date_range("2026-01-01", periods=2, freq="B")))
        self.assertAlmostEqual(float(nav.iloc[-1]), 1.045)


if __name__ == "__main__":
    unittest.main()
