# `optimal_tf` Streamlit Dashboard

Last updated: 2026-05-27

## Purpose

This document covers the local Streamlit dashboard for `optimal_tf`.

It focuses on:
- how to launch the app,
- the main usage modes,
- which services are exposed,
- what outputs are shown in the UI,
- how the dashboard relates to the Python service layer.

See also:
- [optimal_tf_usage.md](/Users/damien.figarol/trading_app_lab/docs/optimal_tf_usage.md) for CLI usage,
- [optimal_tf_architecture.md](/Users/damien.figarol/trading_app_lab/docs/optimal_tf_architecture.md) for package layout and design,
- [optimal_tf_specifications.md](/Users/damien.figarol/trading_app_lab/docs/optimal_tf_specifications.md) for functional scope.

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

The dashboard is organized around 3 usage modes:
- `Standard`
- `Tuning`
- `Inspection`

Each mode exposes one or more services.

### Standard

Purpose:
- use `optimal_tf` in its normal packaged form,
- rely mainly on the TOML config,
- override only a few important parameters when needed.

Available services:
- `Allocation`
- `Evaluation`
- `Compare`

Main standard overrides currently exposed:
- `Universe`
- `Start date`
- `Strategy`
- `Cleaning method`
- `Covariance window`
- `Rebalance frequency` when relevant
- `Long only` when relevant

### Tuning

Purpose:
- run controlled research experiments,
- vary one or several discrete parameters,
- compare scenarios in a research workflow.

Available services:
- `Vary cleaning`
- `Vary window`
- `Vary strategy`
- `Hyperparameter tuning`

Design note:
- `Hyperparameter tuning` is now the generic backend for tuning,
- `Vary cleaning`, `Vary window`, and `Vary strategy` remain available as focused UX shortcuts built on top of that backend.

#### Hyperparameter tuning

This is the most general tuning view.

It can combine:
- several `strategies`,
- several `cleaning methods`,
- several `covariance windows`.

Default behavior:
- if you keep the default UI selections, the dashboard can explore a broad grid,
- the app writes a tabular result set rather than a NAV comparison chart.

Current output in the UI:
- `Results` tab
- `Skipped` tab
- `Config` tab

`Results` shows a compact table centered on:
- `strategy`
- `method`
- `covariance_window`
- main performance metrics such as `sharpe`, `total_return`, `ann_return`, `ann_vol`, `mdd`, `avg_turnover`, `total_cost`, `final_nav`

`Skipped` shows combinations that were intentionally ignored.

Current skip rule:
- `rie*` methods require `covariance_window > number of assets`,
- if that condition is not met, the corresponding scenario is skipped,
- the same window is still evaluated for non-`rie` methods.

Artifacts written by the backend may include:
- `results_table.csv`
- `skipped_configs.csv`
- `request.json`
- `summary.json`

### Inspection

Purpose:
- understand the estimated correlation structure and eigenvectors,
- inspect what the cleaning pipeline is doing,
- analyze factor structure rather than just performance.

Available services:
- `Spectrum by cleaner`
- `Spectrum by window`
- `Eigenvector inspection`

Typical use cases:
- compare eigenvalue spectra across cleaning methods,
- compare eigenvalue spectra across windows,
- inspect selected eigenvectors by sector, sub-sector, or ticker.

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
- multi-strategy and multi-method scenarios use multiselect widgets

This reduces invalid input and keeps the UI aligned with the Python API.

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
- `spectral.py`
  - spectrum analysis services
- `inspection.py`
  - eigenvector inspection services

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
1. adjust or choose a TOML config,
2. launch the dashboard,
3. use `Standard` mode for normal runs,
4. use `Tuning` mode for controlled experiments,
5. use `Inspection` mode when you need to understand the matrix or eigenvectors,
6. keep the generated artifacts for deeper offline analysis if needed.
