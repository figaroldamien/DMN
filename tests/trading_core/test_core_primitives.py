from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import trading_core  # noqa: E402
from market_tickers_data.prices.cache import read_cached_ticker_prices  # noqa: E402
from trading_core.data import load_prices_for_universe, load_prices_yf  # noqa: E402
from trading_core.features import compute_returns, ewma_vol  # noqa: E402
from trading_core.market import get_universe_tickers, list_universes  # noqa: E402
from trading_core.rebalance import resolve_rebalance_dates  # noqa: E402


class TradingCorePrimitiveTests(unittest.TestCase):
    def test_top_level_package_reexports_common_primitives(self) -> None:
        self.assertIs(trading_core.compute_returns, compute_returns)
        self.assertIs(trading_core.load_prices_for_universe, load_prices_for_universe)
        self.assertIs(trading_core.get_universe_tickers, get_universe_tickers)
        self.assertTrue(callable(trading_core.evaluate_portfolio))

    def test_list_universes_includes_recent_equity_sets(self) -> None:
        universes = list_universes()
        self.assertIn("sp500", universes)
        self.assertIn("dji", universes)
        self.assertIn("eurostoxx50", universes)
        self.assertIn("eurostoxx600", universes)
        self.assertIn("sbf120", universes)

    def test_get_universe_tickers_returns_expected_symbols(self) -> None:
        tickers = get_universe_tickers("dji")
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)

    def test_load_prices_yf_renames_share_class_symbols_back_to_requested_names(self) -> None:
        columns = pd.MultiIndex.from_product([["Close"], ["BRK-B", "BF-B"]])
        fake = pd.DataFrame([[10.0, 20.0]], index=[pd.Timestamp("2026-01-02")], columns=columns)
        with patch("market_tickers_data.prices.loader.yf.download", return_value=fake):
            prices = load_prices_yf(["BRK.B", "BF.B"], start="2026-01-01")
        self.assertListEqual(list(prices.columns), ["BRK.B", "BF.B"])

    def test_load_prices_for_universe_delegates_to_market_registry(self) -> None:
        columns = pd.MultiIndex.from_product([["Close"], ["^FCHI"]])
        fake = pd.DataFrame([[100.0]], index=[pd.Timestamp("2026-01-02")], columns=columns)
        with patch("market_tickers_data.prices.loader.yf.download", return_value=fake):
            prices = load_prices_for_universe("test", start="2026-01-01")
        self.assertListEqual(list(prices.columns), ["^FCHI"])

    def test_read_cached_ticker_prices_drops_corrupted_cache_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "BROKEN.parquet"
            cache_path.write_bytes(b"not-a-real-parquet")

            with patch("market_tickers_data.prices.cache.ticker_cache_path", return_value=cache_path):
                series = read_cached_ticker_prices("BROKEN")

            self.assertIsNone(series)
            self.assertFalse(cache_path.exists())

    def test_load_prices_yf_refreshes_when_cached_series_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "BROKEN.parquet"
            metadata_path = Path(tmpdir) / "BROKEN.json"
            cache_path.write_bytes(b"not-a-real-parquet")
            metadata_path.write_text(
                json.dumps(
                    {
                        "ticker": "BROKEN",
                        "provider": "yahoo",
                        "last_fetch_at": "2026-01-10T00:00:00+00:00",
                        "data_start": "2026-01-01",
                        "data_end": "2026-01-10",
                        "rows": 5,
                    }
                ),
                encoding="utf-8",
            )

            downloaded = pd.DataFrame(
                {"BROKEN": [10.0, 11.0]},
                index=pd.to_datetime(["2026-01-09", "2026-01-10"]),
            )
            with patch("market_tickers_data.prices.cache.ticker_cache_path", return_value=cache_path):
                with patch("market_tickers_data.prices.cache.ticker_metadata_path", return_value=metadata_path):
                    with patch("market_tickers_data.prices.loader._download_prices_yf", return_value=downloaded):
                        prices = load_prices_yf(["BROKEN"], start="2026-01-01", refresh_policy="auto")

            self.assertListEqual(list(prices.columns), ["BROKEN"])
            self.assertAlmostEqual(float(prices.iloc[-1, 0]), 11.0)
            self.assertTrue(cache_path.exists())

    def test_compute_returns_and_ewma_vol_produce_frames(self) -> None:
        prices = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [100.0, 99.0, 100.0]})
        returns = compute_returns(prices)
        vol = ewma_vol(returns, span=2)
        self.assertEqual(returns.shape, prices.shape)
        self.assertEqual(vol.shape, prices.shape)

    def test_resolve_rebalance_dates_monthly_uses_last_market_date(self) -> None:
        index = pd.DatetimeIndex(["2026-01-02", "2026-01-30", "2026-02-27"])
        resolved = resolve_rebalance_dates(index, "monthly")
        self.assertListEqual(list(resolved), [pd.Timestamp("2026-01-30"), pd.Timestamp("2026-02-27")])


if __name__ == "__main__":
    unittest.main()
