# `market_dashboard` User Guide

Last updated: 2026-07-16

## Purpose

This document covers the dedicated market synthesis dashboard.

It focuses on:
- how to launch the app,
- how to run market synthesis,
- how to inspect a saved market fork snapshot,
- what sections and outputs the app exposes.

See also:
- [optimal_tf_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/optimal_tf_dashboard.md) for the main portfolio research dashboard,
- [portfolio_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/portfolio_dashboard.md) for fork-based portfolio exploration,
- [optimal_tf_architecture.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_architecture.md) for module boundaries.

## Location

App entrypoint:
- `apps/market_dashboard.py`

Backend service:
- `src/optimal_tf/services/market.py`

Related fork model:
- `src/optimal_tf/market_fork.py`

## Launch

From the repository root:

```bash
cd /Users/damien.figarol/trading_app_lab
PYTHONPATH=src .venv/bin/streamlit run apps/market_dashboard.py
```

## Main Responsibilities

The app does two jobs:
- run a fresh market synthesis for one universe and one date,
- inspect an existing market fork snapshot written by another service.

In practice, it is the market-centric companion to `optimal_tf_dashboard`.

## Sidebar Controls

Main controls:
- `Fork snapshot dir`
- `Recent fork snapshots`
- `Fork snapshot path`
- `Config path`
- `Universe group`
- `Universe`
- `Start`
- `Market date`
- `Output dir`

Operational action:
- `Run market synthesis`

Background utility:
- `Warm all base tickers`

The warm-up tool starts a background cache fill across base tickers so later market loads are less likely to block on first download.

## Result Sections

The app switches between two high-level result families:
- `Synthesis`
- `Fork`

### Synthesis

When a fresh synthesis is available, the UI exposes market views tailored to the current universe type.

Typical tabs include:
- `Overview`
- grouped views such as `Sector`, `Sub-sector`, or `Category`
- `Tickers`
- `Ticker`

Common sub-tabs include:
- `Momentum`
- `Monthly history`
- `NAV`

Typical use cases:
- rank sectors or categories by recent momentum,
- inspect grouped monthly return history,
- inspect equal-weight grouped NAV rebased to 100,
- drill down from a sector or category to individual tickers.

### Fork

When a fork snapshot is loaded, the app displays:
- snapshot metadata,
- source context,
- source request,
- source artifacts.

This is useful when a run from `optimal_tf_dashboard` or another service has already generated a market fork and you want to inspect it without recomputing the market synthesis.

## Parameters Shown In The UI

As with the other dedicated apps, result pages expose:
- `Request payload`
- `Defaults from config`
- `Resolved context`

This is especially helpful when a snapshot path, universe default, or market date comes from preexisting artifacts rather than from the current manual inputs alone.

## Outputs And Artifacts

The service can write artifacts under:
- `output/optimal_tf/market_dashboard`
- fork-oriented outputs under `output/optimal_tf/market_forks`

The exact files depend on the synthesis path, but the app is designed to surface source artifacts when they exist.

## Current Limitations

- the app assumes local execution with access to the same filesystem as the repo,
- large grouped universes can still produce dense tables,
- the warm-up utility is operationally useful but still low-level from a product UX perspective.
