# `optimal_tf` Architecture And Design Notes

Last updated: 2026-03-27

## Purpose

This document is the living design log for the `optimal_tf` project. It tracks:
- architectural choices,
- scope decisions,
- intentionally deferred work,
- how the current modules fit together.

See also:
- [optimal_tf_specifications.md](/Users/damien.figarol/DMN/docs/optimal_tf_specifications.md) for functional requirements,
- [optimal_tf_usage.md](/Users/damien.figarol/DMN/docs/optimal_tf_usage.md) for user-facing commands and config guidance.

Maintenance rule:
- update this file at the end of each working session that changes `optimal_tf`,
- or whenever explicitly requested.

## Project Positioning

`optimal_tf` is a separate project inside the `DMN` repository.

Why:
- the original `dmn` codebase is centered on sequence models and ticker-level ML workflows,
- the new project is centered on cross-asset portfolio construction,
- keeping them separate avoids coupling a portfolio research engine to DMN-specific abstractions.

## High-Level Design

Current design principle:
- separate "estimate weights at a date" from "simulate a portfolio through time".

This leads to two layers:
- allocation layer: compute weights from historical data available up to a date,
- evaluation layer: apply those weights on the following holding period and aggregate performance.

The allocation layer exists, and a first periodic evaluation layer now exists too.

## Current Package Layout

`src/optimal_tf/`

- `config.py`
  Dataclasses for universe, estimation, backtest, and allocation settings.
- `config_io.py`
  Loads TOML config into typed dataclasses.
- `data.py`
  Loads prices from `yfinance` and reuses universes from `market_tickers_data`.
- `features.py`
  Returns, return sanitization, EWMA volatility, volatility-normalized returns, and basic trend signal helpers.
- `estimators/covariance.py`
  Covariance/correlation conversions and PSD repair.
- `estimators/pipeline.py`
  Estimation pipeline:
  prices -> returns -> EWMA vol -> normalized returns -> rolling correlation -> cleaned correlation -> covariance
- `estimators/rie.py`
  Cleaning API. `empirical`, `linear_shrinkage`, and a first native `rie` implementation exist.
- `portfolios.py`
  Current portfolio recipes: `RP` and `ARP`.
- `allocation.py`
  Single-date allocation logic and strategy registry.
- `rebalance.py`
  Rebalance calendar generation from the available market dates.
- `backtest.py`
  Simple weight panel builder and daily return engine.
- `evaluation.py`
  Discrete portfolio simulation with configurable rebalance frequency, turnover, transaction costs and portfolio volatility targeting.
- `metrics.py`
  Basic portfolio statistics plus evaluation summary metrics.
- `reporting.py`
  Benchmark construction and chart rendering for evaluation outputs.
- `validation.py`
  Small comparison helper for native vs reference cleaners.
- `cli/main.py`
  CLI for computing weights at one allocation date.

## Key Design Decisions

### 1. Separate project under the same repo

Decision:
- keep `optimal_tf` under `src/optimal_tf`,
- do not reuse `dmn` internals directly.

Reason:
- the domain model is different,
- the portfolio engine should remain readable and independently testable.

### 2. Reuse `market_tickers_data`

Decision:
- reuse only the market/universe package from the existing repo.

Reason:
- universes are useful shared infrastructure,
- everything else should stay decoupled.

### 3. Estimation and portfolio construction are separate concerns

Decision:
- correlation cleaning belongs to `estimators/`,
- portfolio formulas belong to `portfolios.py`.

Reason:
- `RP`, `ARP`, `NM`, `ToRP` should consume already estimated matrices,
- this keeps the code comparable across estimators.

### 4. Data sanity filtering is applied at the return level

Decision:
- filter implausible single-day returns before estimation, evaluation and benchmark construction.

Reason:
- market data sources can contain broken corporate-action adjustments or unit changes,
- a single bad price point can dominate equal-weight benchmarks and portfolio results.

Current default:
- exclude returns with `abs(return) > 1.0`.

### 5. RIE will be a native implementation

Decision:
- the project will implement RIE internally,
- external implementations will be used only for validation.

Reason:
- the estimator is central to the research workflow,
- we want transparent control over assumptions and numerical choices.

Current state:
- `clean_correlation_matrix(..., method="rie")` now performs a first native nonlinear shrinkage of eigenvalues,
- the next step is validation against an external reference implementation.

### 6. Start with simple, testable portfolio recipes

Decision:
- current implemented recipes are `RP`, `ARP`, `NM`, `EW`, and a first V1 `ToRP`.

Reason:
- they are easier to validate than a full paper-faithful `ToRP` or RIE implementation,
- they give a working end-to-end path for real data integration.

Current note:
- `ToRP` is implemented as a V1 strategy that uses `RP` as the risk budget and a trend signal on volatility-normalized returns as the directional overlay.
- this is intentionally a first operational version, not yet the final paper-faithful implementation.

## Current Defaults

Defaults currently used in config:
- `vol_span = 60`
- `corr_span = 252`
- `corr_min_periods = 252`
- `trend_span = 252`
- `sigma_target_annual = 0.15`
- `cleaning_method = "empirical"` for now

Note:
- the paper target is RIE cleaning, but the code keeps `empirical` as the working default until RIE is implemented.

## CLI Design

Current CLI goal:
- compute portfolio weights at one date for one universe.

Additional CLI goal:
- run a periodic portfolio evaluation over an interval with discrete rebalancing.

Behavior:
- load a universe of tickers,
- fetch price history,
- estimate weights from past data,
- resolve the requested date or the latest available date before it,
- print weights,
- optionally export CSV/JSON.

Current entry points:
- `python -m optimal_tf.cli`
- `optimal-tf`
- `optimal-tf-evaluate`

Packaging:
- the repo now has a `pyproject.toml`,
- editable install works with:
  `/.venv/bin/python -m pip install --no-build-isolation -e .`

## Testing Philosophy

Tests live under `tests/optimal_tf/`, mirroring the package structure.

Current tests cover:
- matrix conversion and PSD repair,
- cleaning baseline behavior,
- portfolio weight normalization,
- basic metrics,
- allocation date resolution,
- rebalance schedule generation,
- periodic evaluation behavior,
- CLI behavior and export output.

## Deferred Work

Not implemented yet:
- external validation and possible refinement of the current RIE implementation,
- refined paper-faithful `ToRP`,
- richer transaction cost models,
- performance reports and plots.

## Next Planned Architecture Extension

Current next major addition should be:
- enrich the evaluation engine,
- add `ToRP`,
- implement and validate RIE.

## Roadmap Snapshot

The most useful next items after the current state are:
- validation/refinement of the native `RIE`,
- anomaly diagnostics export,
- more faithful `ToRP`,
- portfolio combinations,
- richer reporting and benchmark handling.

Immediate next validation loop for `RIE`:
- build a dedicated comparison harness,
- compare `empirical` vs `linear_shrinkage` vs `rie`,
- inspect downstream portfolio behavior on `ARP`, `NM`, and `ToRP`.
