# `portfolio_dashboard` User Guide

Last updated: 2026-07-16

## Purpose

This document covers the dedicated portfolio exploration dashboard.

It focuses on:
- how to load a fork snapshot,
- how to navigate the portfolio by bucket or ticker,
- what equal-weight references are shown,
- what the main tabs mean.

See also:
- [market_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/market_dashboard.md) for market fork inspection,
- [optimal_tf_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/optimal_tf_dashboard.md) for the main `optimal_tf` app,
- [optimal_tf_usage.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/optimal_tf_usage.md) for CLI and export context.

## Location

App entrypoint:
- `apps/portfolio_dashboard.py`

Backend helpers:
- `src/optimal_tf/portfolio_explorer.py`

Input artifact type:
- fork snapshots produced by `optimal_tf_dashboard`

## Launch

From the repository root:

```bash
cd /Users/damien.figarol/trading_app_lab
PYTHONPATH=src .venv/bin/streamlit run apps/portfolio_dashboard.py
```

## Input Model

The app is snapshot-driven.

It expects a fork snapshot path, either:
- chosen from recent snapshots,
- or provided directly through `Fork snapshot path`.

If no snapshot is loaded, the app stays in an informational state and does not attempt to synthesize a portfolio by itself.

## Sidebar Controls

Main controls:
- `Fork snapshot dir`
- `Recent fork snapshots`
- `Fork snapshot path`
- `Return window`

The return window controls the lookback used for summary return comparisons, while the NAV and drawdown views still respect the full available time series inside the snapshot context.

## Main App Tabs

The app exposes 5 main tabs:
- `Portfolio`
- primary bucket tab
- secondary bucket tab when available
- `Ticker`
- `Context`

The primary and secondary bucket tabs depend on the loaded holdings structure:
- sector-first when sector information exists,
- sub-sector as a second level when available,
- ticker-only when no higher hierarchy is available.

## What Each View Shows

### Portfolio

Main content:
- current portfolio summary,
- aggregate exposure view,
- portfolio NAV,
- portfolio drawdown.

This is the fastest way to understand the full portfolio state in the loaded snapshot.

### Bucket Views

For sectors, sub-sectors, or categories, the app shows:
- current holdings in the selected bucket,
- sleeve return summary over the selected lookback,
- NAV comparison versus an equal-weight bucket reference,
- drawdown view,
- positions over time when relevant.

### Ticker

The ticker view drills down to one asset and adds:
- ticker return path,
- equal-weight peer references when available,
- peer tables for sector or sub-sector context.

This makes it easier to answer:
- how this ticker behaved versus its peer bucket,
- whether its contribution is aligned with the sleeve,
- whether it diverged from the equal-weight reference.

### Context

This tab exposes:
- portfolio context,
- source context,
- source request.

Use it when you need to reconnect the explored portfolio to the run that produced it.

## Benchmarks And References

The app uses equal-weight references heavily:
- `EW bucket`
- `EW sector`
- `EW sub-sector`
- `EW universe`

These references are shown only when they are meaningful for the current selection level.

## Current Limitations

- the app depends on an already generated snapshot,
- it is not a portfolio construction UI,
- some grouped selections may still produce dense tables when universes are large.
