import pandas as pd

from .config import EstimationConfig
from .strategies import (
    StrategyPanel,
    StrategyState,
    _lead_lag_virtual_returns,
    compute_portfolio_strategy_state_at_date as _compute_portfolio_strategy_state_at_date,
    compute_portfolio_weights_at_date as _compute_portfolio_weights_at_date,
    compute_strategy_panel as _compute_strategy_panel,
    compute_strategy_state_at_date as _compute_strategy_state_at_date,
    compute_weights_panel as _compute_weights_panel,
    resolve_allocation_date,
    resolve_strategy,
    strategy_registry,
    supported_strategies,
)


def compute_strategy_state_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    date: pd.Timestamp | str,
    long_only: bool = False,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    strategy_context: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> StrategyState:
    return _compute_strategy_state_at_date(
        prices,
        est_cfg,
        strategy,
        date=date,
        long_only=long_only,
        covariance_cache=covariance_cache,
        strategy_context=strategy_context,
    )


def compute_strategy_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    long_only: bool = False,
    target_dates: pd.Index | None = None,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyPanel:
    return _compute_strategy_panel(
        prices,
        est_cfg,
        strategy,
        long_only=long_only,
        target_dates=target_dates,
        covariance_cache=covariance_cache,
    )


def compute_weights_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    long_only: bool = False,
    target_dates: pd.Index | None = None,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return compute_strategy_panel(
        prices,
        est_cfg,
        strategy,
        long_only=long_only,
        target_dates=target_dates,
        covariance_cache=covariance_cache,
    ).effective_weights


def compute_portfolio_weights_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    long_only: bool = False,
) -> tuple[pd.Timestamp, pd.Series]:
    date = resolve_allocation_date(prices.index, as_of_date=as_of_date)
    state = compute_strategy_state_at_date(prices, est_cfg, strategy, date=date, long_only=long_only)
    return date, state.effective_weights.dropna()


def compute_portfolio_strategy_state_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    strategy: str,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    long_only: bool = False,
) -> tuple[pd.Timestamp, StrategyState]:
    date = resolve_allocation_date(prices.index, as_of_date=as_of_date)
    state = compute_strategy_state_at_date(prices, est_cfg, strategy, date=date, long_only=long_only)
    return date, state

__all__ = [
    "StrategyPanel",
    "StrategyState",
    "_lead_lag_virtual_returns",
    "compute_portfolio_strategy_state_at_date",
    "compute_portfolio_weights_at_date",
    "compute_strategy_panel",
    "compute_strategy_state_at_date",
    "compute_weights_panel",
    "resolve_allocation_date",
    "resolve_strategy",
    "strategy_registry",
    "supported_strategies",
]
