from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimal_tf.market_fork import MarketForkSnapshot  # noqa: E402
from optimal_tf.portfolio_explorer import PortfolioExplorerContext, selected_weight_history  # noqa: E402


class PortfolioExplorerTests(unittest.TestCase):
    def test_selected_weight_history_uses_rebalance_dates_for_dynamic_context(self) -> None:
        price_index = pd.date_range("2026-01-01", periods=5, freq="B")
        rebalance_index = pd.DatetimeIndex([price_index[0], price_index[2], price_index[4]])
        prices = pd.DataFrame({"A": [100, 101, 102, 103, 104], "B": [100, 100, 99, 99, 98]}, index=price_index, dtype=float)
        weights_by_rebalance = pd.DataFrame({"A": [0.6, 0.4, 0.5], "B": [0.4, 0.6, 0.5]}, index=rebalance_index, dtype=float)
        daily_weights = pd.DataFrame({"A": [0.0, 0.6, 0.6, 0.4, 0.4], "B": [0.0, 0.4, 0.4, 0.6, 0.6]}, index=price_index, dtype=float)

        context = PortfolioExplorerContext(
            snapshot_path="snapshot.json",
            snapshot=MarketForkSnapshot(source_service="Evaluation", market_universe="test"),
            mode="dynamic",
            universe="test",
            start="2026-01-01",
            as_of_date=price_index[-1],
            anchor_date=rebalance_index[-1],
            trading_start_date=price_index[1],
            prices=prices,
            daily_asset_returns=prices.pct_change().fillna(0.0),
            metadata=pd.DataFrame(index=prices.columns),
            current_weights=weights_by_rebalance.iloc[-1],
            weights_by_rebalance=weights_by_rebalance,
            daily_weights=daily_weights,
            ticker_portfolio_returns=prices.pct_change().fillna(0.0).mul(daily_weights, axis=0),
        )

        history = selected_weight_history(context, ["A", "B"])

        self.assertListEqual(list(history.index), list(rebalance_index))
        self.assertAlmostEqual(float(history.loc[rebalance_index[1], "A"]), 0.4)

    def test_selected_weight_history_keeps_daily_index_for_snapshot_context(self) -> None:
        price_index = pd.date_range("2026-01-01", periods=4, freq="B")
        prices = pd.DataFrame({"A": [100, 101, 102, 103]}, index=price_index, dtype=float)
        snapshot_weights = pd.DataFrame({"A": [0.7]}, index=pd.DatetimeIndex([price_index[1]]), dtype=float)
        daily_weights = pd.DataFrame({"A": [0.0, 0.0, 0.7, 0.7]}, index=price_index, dtype=float)

        context = PortfolioExplorerContext(
            snapshot_path="snapshot.json",
            snapshot=MarketForkSnapshot(source_service="Allocation", market_universe="test"),
            mode="snapshot",
            universe="test",
            start="2026-01-01",
            as_of_date=price_index[-1],
            anchor_date=price_index[1],
            trading_start_date=price_index[2],
            prices=prices,
            daily_asset_returns=prices.pct_change().fillna(0.0),
            metadata=pd.DataFrame(index=prices.columns),
            current_weights=pd.Series({"A": 0.7}),
            weights_by_rebalance=snapshot_weights,
            daily_weights=daily_weights,
            ticker_portfolio_returns=prices.pct_change().fillna(0.0).mul(daily_weights, axis=0),
        )

        history = selected_weight_history(context, ["A"])

        self.assertListEqual(list(history.index), list(price_index))
        self.assertAlmostEqual(float(history.iloc[-1, 0]), 0.7)


if __name__ == "__main__":
    unittest.main()
