from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_core.backtest import EvaluationResult, compare_strategies  # noqa: E402
from trading_core.reporting import EvaluationSummary  # noqa: E402


class CoreComparisonTests(unittest.TestCase):
    def test_compare_strategies_sorts_summary_by_sharpe(self) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 101.0]},
            index=pd.date_range("2026-01-30", periods=3, freq="B"),
        )

        class EvalCfg:
            strategy = "RP"
            rebalance_frequency = "monthly"
            evaluation_start = "2026-01-01"
            evaluation_end = None

        def fake_evaluate(prices_frame, estimation, backtest, eval_cfg):
            scale = 1.1 if eval_cfg.strategy == "ARP" else 1.0
            idx = [pd.Timestamp("2026-01-30")]
            return EvaluationResult(
                summary=EvaluationSummary(
                    total_return=0.1 * scale,
                    ann_return=0.12 * scale,
                    cagr=0.11 * scale,
                    ann_vol=0.15,
                    sharpe=0.8 * scale,
                    sortino=1.0 * scale,
                    skewness=0.1,
                    mar=2.2 * scale,
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
                    total_return_gross=0.101 * scale,
                    total_return_cost_drag=0.001,
                ),
                weights_by_rebalance=pd.DataFrame({"A": [0.6], "B": [0.4]}, index=idx),
                daily_returns_gross=pd.Series([0.0, 0.01 * scale, -0.005], index=prices_frame.index),
                daily_returns_net=pd.Series([0.0, 0.009 * scale, -0.006], index=prices_frame.index),
                turnover_by_rebalance=pd.Series([1.0], index=idx),
                costs_by_rebalance=pd.Series([0.001], index=idx),
                holding_period_returns_gross=pd.Series([0.01 * scale], index=idx),
                holding_period_returns_net=pd.Series([0.009 * scale], index=idx),
            )

        comparison = compare_strategies(
            prices,
            estimation=object(),
            backtest=object(),
            evaluation=EvalCfg(),
            strategies=["RP", "ARP"],
            evaluate_portfolio_fn=fake_evaluate,
        )

        self.assertEqual(list(comparison.summary_table["strategy"]), ["ARP", "RP"])
        self.assertEqual(list(comparison.nav_comparison.columns), ["RP", "ARP"])
        self.assertFalse(comparison.drawdown_comparison.empty)


if __name__ == "__main__":
    unittest.main()
