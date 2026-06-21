from __future__ import annotations

from typing import Literal

import pandas as pd

from ..config import EstimationConfig
from ..strategies.common import resolve_allocation_date, resolve_clean_correlation_at_date
from ..strategies.types import StrategyPanel, StrategyState
from .normalization import apply_long_only_projection, normalize_by_gross_exposure
from .position_engine import build_agnostic_positions
from .q_models import QModel, resolve_q_matrix
from .signals import SignalModel, resolve_signal, resolve_signal_panel

NormalizationMode = Literal["gross", "raw"]


def agnostic_strategy_state_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp | str,
    long_only: bool = False,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    signal_model: SignalModel = "ones",
    q_model: QModel = "identity",
    phi: float = 0.0,
    omega: float = 1.0,
    normalization: NormalizationMode = "gross",
) -> StrategyState:
    """Build one Eq. 8-style strategy state from existing `optimal_tf` inputs."""
    ts = resolve_allocation_date(prices.index, as_of_date=date)
    # ``corr`` already comes from the main estimator pipeline, which cleaned the
    # empirical normalized-return correlation with the configured method.
    corr = resolve_clean_correlation_at_date(prices, est_cfg, ts, covariance_cache=covariance_cache)
    signal = resolve_signal(prices, est_cfg, date=ts, corr=corr, signal_model=signal_model)
    signal_panel = resolve_signal_panel(prices, est_cfg, date=ts, corr=corr, signal_model=signal_model)
    q_matrix = resolve_q_matrix(
        corr,
        q_model=q_model,
        phi=phi,
        est_cfg=est_cfg,
        signal_panel=signal_panel,
    )
    raw = build_agnostic_positions(corr, q_matrix, signal, omega=omega)

    effective = raw.copy()
    if normalization == "gross":
        effective = normalize_by_gross_exposure(effective)
    elif normalization != "raw":
        raise ValueError(f"Unknown normalization mode '{normalization}'.")

    if long_only:
        effective = apply_long_only_projection(effective)

    return StrategyState(
        base_weights=raw.astype(float),
        signal_scale=float(omega),
        effective_weights=effective.astype(float),
    )


def agnostic_strategy_weights_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    date: pd.Timestamp | str,
    long_only: bool = False,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
    signal_model: SignalModel = "ones",
    q_model: QModel = "identity",
    phi: float = 0.0,
    omega: float = 1.0,
    normalization: NormalizationMode = "gross",
) -> pd.Series:
    """Convenience wrapper returning only the effective weights."""
    return agnostic_strategy_state_at_date(
        prices,
        est_cfg,
        date=date,
        long_only=long_only,
        covariance_cache=covariance_cache,
        signal_model=signal_model,
        q_model=q_model,
        phi=phi,
        omega=omega,
        normalization=normalization,
    ).effective_weights


def agnostic_recipe_state_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    recipe_name: str,
    *,
    date: pd.Timestamp | str,
    long_only: bool = False,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyState:
    """Resolve a named agnostic recipe and compute one dated strategy state."""
    from .catalog import resolve_agnostic_recipe

    recipe = resolve_agnostic_recipe(recipe_name)
    return agnostic_strategy_state_at_date(
        prices,
        est_cfg,
        date=date,
        long_only=long_only,
        covariance_cache=covariance_cache,
        signal_model=recipe.signal_model,
        q_model=recipe.q_model,
        phi=recipe.phi,
        omega=recipe.omega,
        normalization=recipe.normalization,
    )


def agnostic_recipe_weights_at_date(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    recipe_name: str,
    *,
    date: pd.Timestamp | str,
    long_only: bool = False,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> pd.Series:
    """Resolve a named agnostic recipe and return only its effective weights."""
    return agnostic_recipe_state_at_date(
        prices,
        est_cfg,
        recipe_name,
        date=date,
        long_only=long_only,
        covariance_cache=covariance_cache,
    ).effective_weights


def compute_agnostic_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    *,
    signal_model: SignalModel,
    q_model: QModel,
    phi: float = 0.0,
    omega: float = 1.0,
    normalization: NormalizationMode = "gross",
    long_only: bool = False,
    target_dates: pd.Index | None = None,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyPanel:
    """Compute a panel of states for one explicit agnostic parameterization."""
    target_index = pd.DatetimeIndex(prices.index if target_dates is None else target_dates)
    base_weights = pd.DataFrame(0.0, index=target_index, columns=prices.columns, dtype=float)
    effective_weights = pd.DataFrame(0.0, index=target_index, columns=prices.columns, dtype=float)
    signal_scale = pd.Series(0.0, index=target_index, dtype=float)

    for ts in target_index:
        try:
            state = agnostic_strategy_state_at_date(
                prices,
                est_cfg,
                date=ts,
                long_only=long_only,
                covariance_cache=covariance_cache,
                signal_model=signal_model,
                q_model=q_model,
                phi=phi,
                omega=omega,
                normalization=normalization,
            )
        except ValueError:
            continue
        base_weights.loc[ts] = state.base_weights.reindex(prices.columns).fillna(0.0)
        effective_weights.loc[ts] = state.effective_weights.reindex(prices.columns).fillna(0.0)
        signal_scale.loc[ts] = float(state.signal_scale)

    return StrategyPanel(
        base_weights=base_weights,
        signal_scale=signal_scale,
        effective_weights=effective_weights,
    )


def compute_agnostic_recipe_panel(
    prices: pd.DataFrame,
    est_cfg: EstimationConfig,
    recipe_name: str,
    *,
    long_only: bool = False,
    target_dates: pd.Index | None = None,
    covariance_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> StrategyPanel:
    """Compute a panel of states for one named agnostic recipe."""
    from .catalog import resolve_agnostic_recipe

    recipe = resolve_agnostic_recipe(recipe_name)
    return compute_agnostic_panel(
        prices,
        est_cfg,
        signal_model=recipe.signal_model,
        q_model=recipe.q_model,
        phi=recipe.phi,
        omega=recipe.omega,
        normalization=recipe.normalization,
        long_only=long_only,
        target_dates=target_dates,
        covariance_cache=covariance_cache,
    )
