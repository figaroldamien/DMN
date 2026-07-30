# `optimal_tf` Architecture And Design Notes

Last updated: 2026-05-27

## Purpose

This document is the living design log for the `optimal_tf` project. It tracks:
- architectural choices,
- scope decisions,
- intentionally deferred work,
- how the current modules fit together.

See also:
- [optimal_tf_specifications.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_specifications.md) for functional requirements,
- [optimal_tf_usage.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/optimal_tf_usage.md) for user-facing commands and config guidance,
- [optimal_tf_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/apps/optimal_tf_dashboard.md) for the local Streamlit UI,
- [optimal_tf_strategies.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_strategies.md) for strategy descriptions and current variants.

Documentation stance:
- `optimal_tf_specifications.md` is now organized as a contract-first document with separate sections for normative requirements, current implementation, and known gaps,
- this architecture note should stay focused on design intent and module boundaries rather than repeating the functional contract.

Maintenance rule:
- update this file at the end of each working session that changes `optimal_tf`,
- or whenever explicitly requested.

## Project Positioning

`optimal_tf` is a separate application inside the repository.

Current repository direction:
- the long-term root name target is `trading_app_lab`,
- shared trading infrastructure is progressively moving into `trading_core`,
- `optimal_tf` is now the main reference application for that shared layer.

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

Current implementation detail:
- the primary primitive is now date-centric: allocation is computed for one resolved market date at a time,
- periodic evaluation loops over rebalance dates,
- the evaluation engine precomputes one covariance cache for the run and reuses it across rebalance dates to avoid recomputing the full estimator history at each step,
- the standard covariance estimator now uses a fixed historical window plus cleaning rather than an EWMA covariance smoother.

Timing convention carried by the current design:
- rebalance weights are observed on date `t`,
- they are applied only to returns strictly after `t`,
- the next rebalance date is the end of the holding period and is included in that period's return stream,
- rebalance cost is charged on the first trading day strictly after `t`.

## Current Package Layout

`src/optimal_tf/`

- `config.py`
  Dataclasses for universe, estimation, backtest, and allocation settings.
- `config_io.py`
  Loads TOML config into typed dataclasses.
- `data.py`
  Compatibility facade over shared data loading in `trading_core.data`.
- `features.py`
  Compatibility facade for shared feature primitives plus local rolling correlation helpers still used by the legacy estimator path.
- `estimators/`
  Compatibility facades over shared risk code in `trading_core.risk`.
- `portfolios.py`
  Current portfolio recipes: `RP` and `ARP`.
- `allocation.py`
  Compatibility facade exposing the public allocation API.
- `strategies/`
  Strategy package split by concern:
  `base.py`, `torp.py`, `lltf.py`, `common.py`, `types.py`, `api.py`.
- `strategies_agnostic/`
  Experimental Eq. 8 strategy lab:
  `position_engine.py`, `q_models.py`, `signals.py`, `normalization.py`, `catalog.py`, `api.py`.
- `rebalance.py`
  Compatibility facade over shared rebalance helpers in `trading_core.rebalance`.
- `backtest.py`
  Legacy/simple weight panel builder plus daily return engine utilities.
- `evaluation.py`
  Compatibility facade exposing the public evaluation API.
- `eval/`
  Compatibility facades over shared periodic evaluation types and engine in `trading_core.backtest`.
- `metrics.py`
  Compatibility facade over shared reporting metrics in `trading_core.reporting`.
- `reporting.py`
  Compatibility facade exposing shared benchmark helpers and plotting functions from `trading_core.reporting`.
- `plots/`
  Plot and benchmark package split into `benchmarks.py` and `renderers.py`.
- `validation.py`
  Small comparison helper for native vs reference cleaners.
- `cli/main.py`
  CLI for computing weights at one allocation date.
- `cli/evaluate.py`
  CLI for periodic evaluation, exports, and benchmark chart generation.
- `cli/compare.py`
  CLI for multi-strategy comparison runs, comparison tables, and first comparison plots.

`src/trading_core/`

- `market/`
  Shared universe access.
- `data/`
  Shared price loading and symbol adaptation.
- `features/`
  Shared return, volatility, trend, and transform primitives.
- `risk/`
  Shared covariance and cleaning pipeline.
- `rebalance/`
  Shared rebalance date helpers.
- `backtest/`
  Shared periodic evaluation and multi-strategy comparison engine.
- `reporting/`
  Shared metrics, plots, benchmarks, and export helpers.

## Key Design Decisions

### 1. Separate application on top of a shared core

Decision:
- keep `optimal_tf` under `src/optimal_tf`,
- keep shared quant mechanics in `src/trading_core`,
- do not reuse `dmn` internals directly.

Reason:
- the domain model is different,
- the portfolio engine should remain readable and independently testable,
- future trading apps should reuse a common infrastructure layer instead of copying `optimal_tf`.

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
- `RP`, `ARP`, `NM`, `ToRP0`, `ToRP1`, `ToRP2`, and `ToRP3` should consume already estimated matrices,
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
- current implemented recipes are `RP`, `ARP`, `NM`, `EW`, `ToRP0`, `ToRP1`, `ToRP2`, and `ToRP3`.

Reason:
- they are easier to validate than a full paper-faithful `ToRP` or RIE implementation,
- they give a working end-to-end path for real data integration.

Current note:
- `LLTF` is an empirical lead-lag trend-following strategy built from EWMA moments of virtual cross-asset streams `r_j * s_k`.
- `ToRP0` is the historical implementation: an asset-by-asset trend overlay on the `RP` budget.
- `ToRP1` measures a common trend signal on the `RP` factor and applies that signal back onto the `RP` portfolio.
- `ToRP2` computes the trend on the `RP` factor return stream itself and applies that factor signal back onto a category-aware `RP` portfolio.
- `ToRP3` keeps the trend amplitude explicit through `signal_scale` and `effective_weights`.
- `ToRP3` is the closest current implementation to Sec. 3.4 of the reference paper, though the framework still layers a separate portfolio volatility target on top.
- `strategies_agnostic` is the new experimental Eq. 8 path used to compare agnostic recipes before deciding whether any should migrate into the main router.

Experimental agnostic policy:
- empirical `C` is cleaned by the standard estimator pipeline with the configured method,
- structural `Q` models (`I`, `C`, `Q_phi`) are only stabilized structurally, not statistically re-shrunk,
- future empirical `Q` models may later opt into the same configurable cleaning family as `C`.

Performance note:
- after the move to date-centric allocation, the evaluation engine now reuses a covariance cache within one run,
- this materially improves execution time for `RP`, `ARP`, `NM`, and the `ToRP` family on moderate universes.
- `ToRP2` and `ToRP3` now also reuse a precomputed `RP` factor context within one evaluation run, so they no longer rebuild the full historical `RP` path at each rebalance date.

Documentation consequence:
- strategy names are now part of the public contract and should remain stable once exposed through config and CLI.

## Current Defaults

Defaults currently used in config:
- `vol_span = 60`
- `covariance_window = 252`
- `covariance_min_periods = 252`
- `trend_alpha = 0.01`
- `torp_signal_gain = 5.0`
- `sigma_target_annual = 0.15`
- `cost_bps = 25.0`
- `weight_smoothing_alpha = 1.0`
- `long_only = true`
- `cleaning_method = "rie"`
- reference benchmark-only cleaner also available as `cleaning_method = "rie_reference"` when the optional `rie-estimator` package is installed
- `allocation.strategy = "ARP"`
- `evaluation.strategy = "ARP"`
- `universe.name = "world_index"`

Note:
- the dataclass defaults are still more conservative than the example config in a few places,
- the example config is the best reference for the current recommended working setup.

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
- richer transaction cost models,
- richer performance reports and plots beyond the current baseline chart export.

## Next Planned Architecture Extension

Current next major addition should be:
- enrich the evaluation engine,
- extend `ToRP` variants,
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
- inspect downstream portfolio behavior on `ARP`, `NM`, `ToRP0`, `ToRP1`, `ToRP2`, and `ToRP3`.
