# `matrix_inspection_dashboard` User Guide

Last updated: 2026-07-16

## Purpose

This document covers the dedicated matrix inspection dashboard.

It focuses on:
- the workspace/config model,
- the currently exposed inspection services,
- the structure of each result view,
- how the app differs from the main `optimal_tf_dashboard`.

See also:
- [optimal_tf_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/optimal_tf_dashboard.md) for the main portfolio research dashboard,
- [optimal_tf_architecture.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_architecture.md) for service-layer design,
- [optimal_tf_usage.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/optimal_tf_usage.md) for repo workflow notes.

## Location

App entrypoint:
- `apps/matrix_inspection_dashboard.py`

Dedicated config:
- `configs/matrix_inspection.toml`

Backend services:
- `src/optimal_tf/services/inspection.py`

## Launch

From the repository root:

```bash
cd /Users/damien.figarol/trading_app_lab
PYTHONPATH=src .venv/bin/streamlit run apps/matrix_inspection_dashboard.py
```

## App Structure

The app is organized around 2 usage modes:
- `Workspace`
- `Inspection`

### Workspace

Service:
- `Config editor`

Purpose:
- edit the dedicated matrix inspection TOML directly from the UI,
- keep persistent defaults for snapshot, interval and core-periphery runs in one app-specific config.

The config editor covers:
- universe defaults,
- estimation defaults,
- evaluation defaults,
- app defaults for snapshot, interval and core-periphery outputs.

### Inspection

Current services:
- `Inspect at date`
- `Core-periphery at date`
- `Inspect over interval`

## Shared Sidebar Context

Outside `Workspace`, the app exposes shared controls for:
- `Config path`
- `Universe group`
- `Universe`
- `Start date`
- `Evaluation start`
- `Evaluation end`
- `Refresh prices now`

These controls define the workspace context reused by the run-specific forms.

## Service Details

### Inspect at date

Purpose:
- inspect one dated cleaned-matrix state in depth.

Main form controls:
- matrix type,
- input type,
- estimator,
- estimator window,
- cleaning method,
- linear shrinkage,
- inspection date,
- output dir.

Result tabs:
- `Summary`
- `Matrices`
- `Spectrum`
- `Eigenvectors`
- `Config`
- `Artifacts`

Notable matrix views:
- cleaned matrix heatmap,
- cleaner delta heatmap,
- sector EW correlation,
- intra-sector sub-sector EW correlations.

### Core-periphery at date

Purpose:
- compute per-ticker coreness from the cleaned correlation graph on one date.

Main form controls:
- input type,
- estimator,
- graph filter,
- estimator window,
- cleaning method,
- linear shrinkage,
- inspection date,
- output dir.

Graph filter options:
- `full_graph`
- `mst`

Result tabs:
- `Summary`
- `Ranking`
- `Graph`
- `Matrices`
- `Config`
- `Artifacts`

Current graph views:
- color by sector,
- color by coreness.

Current graph behavior:
- node size and border width reflect coreness,
- in `full_graph`, the graph view exposes a `Max displayed edges` slider,
- the default display heuristic keeps roughly 3 edges per ticker, with a floor at 60, capped by the number of available non-zero edges.

### Inspect over interval

Purpose:
- study how the cleaned matrix spectrum and leading eigenmodes evolve over a rebalance interval.

Main form controls:
- matrix type,
- input type,
- estimator,
- estimator window,
- cleaning method,
- linear shrinkage,
- inspection frequency,
- leading eigenvectors,
- output dir.

Result tabs:
- `Summary`
- `Spectrum trends`
- `Eigenvector stability`
- `Config`
- `Artifacts`

Spectrum trends includes:
- leading eigenvalues over time,
- Marchenko-Pastur outlier counts above `lambda+` and below `lambda-`,
- eigenvalue variogram.

## Parameters Shown In The UI

Result pages expose:
- `Request payload`
- `Defaults from config`
- `Resolved context`

This is especially useful because the app blends:
- dedicated config defaults,
- shared sidebar workspace context,
- run-specific overrides.

## Current Limitations

- graph rendering depends on `pyvis` for the interactive core-periphery view,
- some dense universes still require edge filtering to keep the graph readable,
- the app is inspection-first and not intended to replace the main portfolio evaluation workflow.
