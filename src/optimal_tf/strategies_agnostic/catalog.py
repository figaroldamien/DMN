from __future__ import annotations

from dataclasses import dataclass

from .api import NormalizationMode, QModel, SignalModel


@dataclass(frozen=True)
class AgnosticRecipe:
    name: str
    signal_model: SignalModel
    q_model: QModel
    phi: float = 0.0
    omega: float = 1.0
    normalization: NormalizationMode = "gross"


AGNOSTIC_RECIPE_REGISTRY: dict[str, AgnosticRecipe] = {
    "ARP_AGNOSTIC": AgnosticRecipe(
        name="ARP_AGNOSTIC",
        signal_model="ones",
        q_model="identity",
        normalization="gross",
    ),
    "MARKOWITZ_AGNOSTIC": AgnosticRecipe(
        name="MARKOWITZ_AGNOSTIC",
        signal_model="ones",
        q_model="correlation",
        normalization="gross",
    ),
    "ATF_AGNOSTIC": AgnosticRecipe(
        name="ATF_AGNOSTIC",
        signal_model="trend_ema",
        q_model="identity",
        normalization="gross",
    ),
    "ATF_RAW": AgnosticRecipe(
        name="ATF_RAW",
        signal_model="trend_ema",
        q_model="identity",
        normalization="raw",
    ),
    "ATF_EMPIRICAL_Q": AgnosticRecipe(
        name="ATF_EMPIRICAL_Q",
        signal_model="trend_ema",
        q_model="empirical",
        normalization="gross",
    ),
    "PHI_0": AgnosticRecipe(
        name="PHI_0",
        signal_model="ones",
        q_model="phi_shrink_correlation",
        phi=0.0,
        normalization="gross",
    ),
    "PHI_25": AgnosticRecipe(
        name="PHI_25",
        signal_model="ones",
        q_model="phi_shrink_correlation",
        phi=0.25,
        normalization="gross",
    ),
    "PHI_50": AgnosticRecipe(
        name="PHI_50",
        signal_model="ones",
        q_model="phi_shrink_correlation",
        phi=0.5,
        normalization="gross",
    ),
    "PHI_100": AgnosticRecipe(
        name="PHI_100",
        signal_model="ones",
        q_model="phi_shrink_correlation",
        phi=1.0,
        normalization="gross",
    ),
}


def agnostic_recipe_registry() -> dict[str, AgnosticRecipe]:
    """Return a shallow copy of the named agnostic recipe catalog."""
    return dict(AGNOSTIC_RECIPE_REGISTRY)


def supported_agnostic_strategies() -> list[str]:
    """Return the sorted list of recipe names exposed by the agnostic lab."""
    return sorted(AGNOSTIC_RECIPE_REGISTRY)


def resolve_agnostic_recipe(name: str) -> AgnosticRecipe:
    """Resolve one named agnostic recipe."""
    if name not in AGNOSTIC_RECIPE_REGISTRY:
        raise KeyError(f"Unknown agnostic strategy '{name}'. Allowed values: {sorted(AGNOSTIC_RECIPE_REGISTRY)}")
    return AGNOSTIC_RECIPE_REGISTRY[name]
