from __future__ import annotations

from typing import Final

PRODUCT_MODES: Final[tuple[str, ...]] = (
    "Workspace",
    "Run",
    "Compare",
    "Search",
    "Guide",
)

MODE_INTRO: Final[dict[str, str]] = {
    "Workspace": "Configure the shared research workspace before running analyses.",
    "Run": "Execute one focused analysis or diagnostic under the current workspace context.",
    "Compare": "Compare controlled alternatives under shared assumptions in one comparison family.",
    "Search": "Explore broader strategy and parameter spaces to find promising candidates.",
    "Guide": "Use the guide area to understand strategies and service roles before running experiments.",
}

MODE_SERVICES: Final[dict[str, dict[str, str]]] = {
    "Workspace": {
        "Config editor": "Edit the TOML configuration used by optimal_tf directly from the dashboard.",
    },
    "Run": {
        "Allocation": "Single-date allocation of one strategy.",
        "Evaluation": "Packaged backtest for one strategy over an evaluation window.",
        "Matrix inspection": "Inspect one dated cleaned-matrix state with spectra, eigenvectors and cross-asset features.",
    },
    "Compare": {
        "Compare": "Compare several strategies under one shared market and backtest context.",
        "Vary strategy": "Compare strategies for one cleaner and one covariance window.",
        "Vary cleaning": "Compare correlation cleaning methods for one strategy.",
        "Vary window": "Compare covariance lookback windows for one strategy and one cleaner.",
        "Vary frequency": "Compare rebalance frequencies for one strategy, one cleaner and one covariance window.",
    },
    "Search": {
        "Strategy testbed": "Focused agnostic strategy sandbox with explicit signal, Q, phi, omega and normalization controls.",
        "Hyperparameter tuning": "Grid search over strategies, cleaning methods, covariance windows and rebalance frequencies.",
    },
    "Guide": {
        "Strategy guide": "Short guide describing the currently exposed strategies.",
    },
}

SERVICE_INTRO: Final[dict[tuple[str, str], str]] = {
    ("Workspace", "Config editor"): "Administrative view to inspect and update the shared TOML defaults used by the dashboard.",
    ("Run", "Allocation"): "Single-date allocation service. Use it when you want the latest portfolio weights rather than a full backtest.",
    ("Run", "Evaluation"): "Packaged backtest service. Use it when you want performance, turnover and benchmark comparison over an evaluation window.",
    ("Run", "Matrix inspection"): "Diagnostic snapshot of one dated cleaned-matrix state with spectra, eigenvectors and cross-asset features.",
    ("Compare", "Compare"): "Comparison lab entry point for multi-strategy analysis under one shared market and backtest context.",
    ("Compare", "Vary strategy"): "Comparison experiment that keeps the market context fixed and compares several strategy families under the same estimation settings.",
    ("Compare", "Vary cleaning"): "Comparison experiment that keeps the strategy fixed and compares several cleaning methods under the same evaluation setup.",
    ("Compare", "Vary window"): "Comparison experiment that keeps strategy and cleaning fixed while testing several covariance lookback windows.",
    ("Compare", "Vary frequency"): "Comparison experiment that keeps the strategy fixed and compares multiple rebalance frequencies as an operational trade-off study.",
    ("Search", "Strategy testbed"): "Research sandbox for one strategy configuration with explicit control over signal, Q, phi, normalization and execution settings.",
    ("Search", "Hyperparameter tuning"): "Advanced search view that evaluates a grid of strategies, cleaning methods, covariance windows and rebalance frequencies.",
    ("Guide", "Strategy guide"): "Reference page for the exposed strategy families and their practical meaning before running a service.",
}

COMMON_EVALUATION_DATE_SERVICES: Final[set[tuple[str, str]]] = {
    ("Run", "Evaluation"),
    ("Run", "Matrix inspection"),
    ("Compare", "Compare"),
    ("Compare", "Vary strategy"),
    ("Compare", "Vary cleaning"),
    ("Compare", "Vary window"),
    ("Compare", "Vary frequency"),
    ("Search", "Strategy testbed"),
    ("Search", "Hyperparameter tuning"),
}


def all_service_routes() -> list[tuple[str, str]]:
    return [
        (mode, service_name)
        for mode, services in MODE_SERVICES.items()
        for service_name in services
    ]


def mode_for_service(service_name: str) -> str | None:
    for mode, services in MODE_SERVICES.items():
        if service_name in services:
            return mode
    return None


def compare_service_names() -> list[str]:
    return list(MODE_SERVICES["Compare"])
