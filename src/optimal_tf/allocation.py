import pandas as pd

from .config import EstimationConfig
from .strategies_agnostic import (
    agnostic_recipe_state_at_date as _agnostic_recipe_state_at_date,
    compute_agnostic_recipe_panel as _compute_agnostic_recipe_panel,
    supported_agnostic_strategies,
)
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
    supported_strategies as _supported_legacy_strategies,
)


def supported_strategies() -> list[str]:
    """Return the full set of dashboard/CLI-visible strategy identifiers.

    This now includes both the legacy `optimal_tf.strategies` family and the
    experimental Eq. 8 agnostic recipes, so service and UI layers can expose a
    unified strategy selector.
    """
    return sorted({*_supported_legacy_strategies(), *supported_agnostic_strategies()})


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
    if strategy in supported_agnostic_strategies():
        return _agnostic_recipe_state_at_date(
            prices,
            est_cfg,
            strategy,
            date=date,
            long_only=long_only,
            covariance_cache=covariance_cache,
        )
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
    if strategy in supported_agnostic_strategies():
        return _compute_agnostic_recipe_panel(
            prices,
            est_cfg,
            strategy,
            long_only=long_only,
            target_dates=target_dates,
            covariance_cache=covariance_cache,
        )
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
