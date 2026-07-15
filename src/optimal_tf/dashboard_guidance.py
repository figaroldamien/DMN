from __future__ import annotations

from typing import Any, Mapping


def workspace_overview_rows(config_path: str, config_defaults: Mapping[str, Any]) -> list[dict[str, str]]:
    universe_defaults = config_defaults.get("universe", {})
    evaluation_defaults = config_defaults.get("evaluation", {})
    compare_defaults = config_defaults.get("compare", {})
    allocation_defaults = config_defaults.get("allocation", {})
    compare_strategies = list(compare_defaults.get("strategies") or [])
    return [
        {"field": "config_path", "value": str(config_path)},
        {"field": "universe", "value": str(universe_defaults.get("name", ""))},
        {"field": "start", "value": str(universe_defaults.get("start", ""))},
        {
            "field": "evaluation_window",
            "value": _join_non_empty(
                [
                    _optional_str(evaluation_defaults.get("evaluation_start")),
                    _optional_str(evaluation_defaults.get("evaluation_end")),
                ],
                separator=" -> ",
            ),
        },
        {"field": "allocation_strategy", "value": str(allocation_defaults.get("strategy", ""))},
        {"field": "evaluation_strategy", "value": str(evaluation_defaults.get("strategy", ""))},
        {"field": "compare_scope", "value": f"{len(compare_strategies)} strategies" if compare_strategies else ""},
    ]


def guide_service_choices() -> list[dict[str, str]]:
    return [
        {
            "service_family": "Run",
            "when_to_use": "You want one concrete answer for one strategy under one shared context.",
            "best_for": "Allocation snapshot and packaged evaluation.",
        },
        {
            "service_family": "Compare",
            "when_to_use": "You want to hold most assumptions fixed and vary one dimension.",
            "best_for": "Strategies, cleaning methods, covariance windows, rebalance frequencies.",
        },
        {
            "service_family": "Search",
            "when_to_use": "You want to explore a wider design space before deciding what to keep.",
            "best_for": "Focused sandbox in Strategy testbed, broader grid in Hyperparameter tuning.",
        },
    ]


def guide_next_step_rows() -> list[dict[str, str]]:
    return [
        {
            "goal": "I need current weights",
            "recommended_service": "Run / Allocation",
            "why": "Best when you want one dated portfolio output rather than a performance path.",
        },
        {
            "goal": "I need a benchmarked backtest",
            "recommended_service": "Run / Evaluation",
            "why": "Best when you want NAV, turnover, costs and benchmark-relative performance.",
        },
        {
            "goal": "I want to inspect one dated state in depth",
            "recommended_service": "Dedicated app / matrix inspection dashboard / Inspect at date",
            "why": "Best when you need cleaned matrices, eigen-structure and cross-asset diagnostics outside the main portfolio app.",
        },
        {
            "goal": "I want to see how eigenmodes evolve over time",
            "recommended_service": "Dedicated app / matrix inspection dashboard / Inspect over interval",
            "why": "Best when you want the spectrum and leading eigenvector stability across a rebalance interval.",
        },
        {
            "goal": "I want to compare a few controlled alternatives",
            "recommended_service": "Compare / Comparison Lab",
            "why": "Best when one dimension varies and the rest should stay shared.",
        },
        {
            "goal": "I want to shape a strategy recipe interactively",
            "recommended_service": "Search / Strategy testbed",
            "why": "Best focused sandbox for signals, Q choices, phi, normalization and cleaning.",
        },
        {
            "goal": "I want to search more broadly",
            "recommended_service": "Search / Hyperparameter tuning",
            "why": "Best when you are ready to evaluate a grid instead of a single main scenario.",
        },
    ]


def _join_non_empty(parts: list[str | None], *, separator: str) -> str:
    values = [part for part in parts if part]
    return separator.join(values)


def _optional_str(value: Any) -> str | None:
    if value in (None, "", "None"):
        return None
    return str(value)
