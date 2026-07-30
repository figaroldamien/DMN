# `optimal_tf` Streamlit Dashboard

Last updated: 2026-07-16

## Purpose

This document covers the local Streamlit dashboard for `optimal_tf`.

It focuses on:
- how to launch the app,
- the main usage modes,
- which services are exposed,
- what outputs are shown in the UI,
- how the dashboard relates to the Python service layer.

See also:
- [optimal_tf_usage.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/optimal_tf_usage.md) for CLI usage,
- [optimal_tf_architecture.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_architecture.md) for package layout and design,
- [optimal_tf_specifications.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_specifications.md) for functional scope,
- [market_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/market_dashboard.md) for market synthesis and fork inspection,
- [portfolio_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/portfolio_dashboard.md) for fork-based portfolio exploration,
- [matrix_inspection_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/matrix_inspection_dashboard.md) for matrix and core-periphery diagnostics.

## Location

Dashboard entrypoint:
- `apps/optimal_tf_dashboard.py`

Backend service layer used by the dashboard:
- `src/optimal_tf/services/`

Important note:
- the dashboard does **not** call the CLI,
- it calls the Python service layer directly,
- this is intentional so the same backend can later be reused by a local UI, a server API, or a React frontend.

## Launch

From the repository root:

```bash
cd /Users/damien.figarol/trading_app_lab
PYTHONPATH=src .venv/bin/streamlit run apps/optimal_tf_dashboard.py
```

If needed, install the project in editable mode first:

```bash
cd /Users/damien.figarol/trading_app_lab
.venv/bin/python -m pip install --no-build-isolation -e .
```

## App Structure

The dashboard is organized around 5 usage modes:
- `Workspace`
- `Run`
- `Compare`
- `Search`
- `Guide`

Each mode exposes one or more services.

### Workspace

Purpose:
- edit the shared TOML workspace directly from the UI,
- make persistent changes to defaults without leaving Streamlit.

Available services:
- `Config editor`

What this page changes:
- universe defaults,
- evaluation window defaults,
- strategy defaults,
- output path defaults.

What it does not do:
- it does not run a backtest,
- it does not produce artifacts by itself.

### Guide

Purpose:
- provide a quick in-app orientation page,
- describe the currently exposed strategies in a few words,
- reduce the need to leave the dashboard to remember what each strategy means.

Available services:
- `Strategy guide`

Note:
- market synthesis has been moved out of `optimal_tf_dashboard`,
- use `apps/market_dashboard.py` for market-specific views.

### Run

Purpose:
- use `optimal_tf` in its normal packaged form,
- rely mainly on the TOML config,
- override only a few important parameters when needed.

Current services:
- `Allocation`
- `Evaluation`

Main run overrides currently exposed:
- `Universe`
- `Start date`
- `Strategy`
- `Cleaning method`
- `Covariance window`
- `Rebalance frequency` when relevant
- `Long only` when relevant

### Compare

Purpose:
- compare controlled alternatives under one shared market and backtest context.

- Current services:
- `Compare`
- `Vary cleaning`
- `Vary window`
- `Vary strategy`
- `Vary frequency`

Typical use:
- hold most assumptions fixed,
- vary one dimension deliberately,
- compare outputs in one comparison family.

### Search

Purpose:
- explore broader strategy or parameter spaces,
- search before narrowing down to one comparison or one operational run.

Current services:
- `Strategy testbed`
- `Hyperparameter tuning`

#### Strategy testbed

Purpose:
- focused sandbox for one strategy configuration,
- explicit controls for signal, `Q`, `phi`, normalization and execution assumptions.

Typical use:
- shape one agnostic recipe interactively,
- inspect one candidate before moving to comparison or search.

#### Hyperparameter tuning

This is the broadest search view.

It can combine:
- several `strategies`,
- several `cleaning methods`,
- several `covariance windows`,
- several `rebalance frequencies`.

Default behavior:
- it evaluates a grid rather than a single main scenario,
- the main UI result is tabular rather than a single NAV comparison chart.

Current output in the UI:
- `Results`
- `Skipped`
- `Config`

Current skip rule:
- `rie*` methods require `covariance_window > number of assets`,
- if that condition is not met, the corresponding scenario is skipped,
- the same window is still evaluated for non-`rie` methods.

Artifacts written by the backend may include:
- `results_table.csv`
- `skipped_configs.csv`
- `request.json`
- `summary.json`

## Parameters Shown In The UI

A core dashboard rule is now:
- the user should not need to reopen the TOML file just to understand what was run.

For that reason, result pages show a `Parameters used` section with 3 views:
- `Request payload`
- `Defaults from config`
- `Resolved context`

Meaning:
- `Request payload`
  what the UI sent to the Python service layer,
- `Defaults from config`
  what the current TOML file contains,
- `Resolved context`
  what the service effectively used or resolved for this run.

This is especially useful for:
- `strategy`
- `cleaning_method`
- `covariance_window`
- `rebalance_frequency`
- scenario counts and skipped combinations in tuning mode

## Enumerated Inputs

The dashboard uses guided widgets where possible instead of raw text fields.

Current examples:
- `Universe` uses a dropdown built from `market_tickers_data.MARKET_TICKERS`
- `Strategy` uses the list from `supported_strategies()`
- `Cleaning method` uses the list from `supported_cleaning_methods()`
- `Rebalance frequency` uses the list from `supported_rebalance_frequencies()`
- multi-strategy scenarios now use main-page checkbox selectors grouped by family

This reduces invalid input and keeps the UI aligned with the Python API.

Current strategy selector behavior:

- single-strategy services use the same grouped selector layout as the
  multi-strategy services, but with exclusive checkboxes,
- the selector is split into two blocks:
  - `Classiques`
  - agnostic families exposed by the app
- the agnostic block is intentionally wider because the recipe names are longer,
- examples of exposed agnostic recipes include `ARP_AGNOSTIC`,
  `MARKOWITZ_AGNOSTIC`, `ATF_AGNOSTIC`, `PHI_25`, and `PHI_50`,
- redundant endpoint aliases such as `PHI_0` and `PHI_100` are kept in code for
  research scripts but hidden from the dashboard to reduce UI clutter,
- the dashboard still treats them as experimental research strategies rather
  than as a replacement for the legacy packaged defaults.
- the strategy help now lives in its own `Guide` mode instead of being shown on
  every service page.

## Relationship With The Service Layer

The dashboard is a thin interactive frontend over `src/optimal_tf/services/`.

Current service groups:
- `standard.py`
  - `run_allocation`
  - `run_evaluation`
  - `run_compare`
- `evaluation.py`
  - `run_hyperparameter_tuning`
  - `run_vary_cleaning`
  - `run_vary_window`
  - `run_vary_strategy`
  - `run_vary_frequency`
- `spectral.py`
  - spectrum analysis services
- `inspection.py`
  - matrix and eigenvector diagnostics exposed through the dedicated matrix app
- `market.py`
  - market synthesis backend used by the dedicated market app

Design consequence:
- the dashboard should stay mostly declarative,
- behavior changes should happen primarily in the service layer,
- the same backend can later be reused by a server API.

## Current Limitations

The dashboard is still a research UI, not a polished product UI.

Known limitations:
- some large tables may still need horizontal space on smaller screens,
- chart layout is functional but not yet heavily optimized for dense scenario exploration,
- some advanced research outputs still remain easier to inspect from generated CSV files,
- the dashboard assumes local execution in the same environment as the repo.

## Recommended Workflow

A practical workflow is:
- set persistent defaults in `Workspace / Config editor`,
- use `Guide / Strategy guide` if you need orientation,
- use `Run / Allocation` or `Run / Evaluation` for one concrete answer,
- use `Compare` when one dimension should vary under shared assumptions,
- use `Search` when you want to explore a wider design space before narrowing down,
- switch to the dedicated apps when the task becomes market-specific, fork-specific, or matrix-specific.
1. adjust or choose a TOML config,
2. launch the dashboard,
3. use `Standard` mode for normal runs,
4. use `Tuning` mode for controlled experiments,
5. use `Inspection` mode when you need to understand the matrix or eigenvectors,
6. keep the generated artifacts for deeper offline analysis if needed.
