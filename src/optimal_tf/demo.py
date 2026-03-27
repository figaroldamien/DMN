from __future__ import annotations

from .backtest import backtest_portfolio
from .config import BacktestConfig, EstimationConfig, UniverseConfig
from .data import load_prices_for_universe
from .metrics import performance_metrics
from .portfolios import risk_parity_weights_from_cov


def run_demo() -> None:
    universe = UniverseConfig(name="test", start="2020-01-01")
    est_cfg = EstimationConfig(cleaning_method="empirical")
    bt_cfg = BacktestConfig(long_only=True)
    prices = load_prices_for_universe(universe.name, start=universe.start)
    pnl, turnover, _ = backtest_portfolio(prices, est_cfg, bt_cfg, risk_parity_weights_from_cov)
    perf = performance_metrics(pnl, turnover)
    print(f"Sharpe={perf.sharpe:.3f} ann_return={perf.ann_return:.3f} ann_vol={perf.ann_vol:.3f}")
