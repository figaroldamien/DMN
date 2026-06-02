# ADR 0002: Prepare Root Repository Rename To `trading_app_lab`

Date: 2026-05-12

## Status

Executed

## Context

The repository root was named `DMN`, while the project has evolved into a
broader trading research workspace containing:
- `optimal_tf`
- shared infrastructure in `trading_core`
- market universe data in `market_tickers_data`
- legacy `dmn` code that is no longer the main architectural driver

Keeping `DMN` as the root name understated the actual scope of the repo and
created confusion in docs and onboarding.

## Decision

The target root repository name became:
- `trading_app_lab`

The rename was intentionally deferred until after the shared extraction work was
stable enough, so we did not mix:
- import and module refactors
- doc updates
- path changes
- repo rename noise

## Consequences

Benefits:
- better alignment between repository name and actual purpose
- clearer onboarding for future trading research apps
- less coupling between the repo identity and the legacy `dmn` app

Trade-offs:
- some historical ADR text still refers to the former root name for context
- local path examples still need routine cleanup as docs evolve

## Rename Readiness Checklist

The root rename was considered ready once all of the following were true:

1. `trading_core` covers the shared market, data, feature, risk, backtest, and
   reporting layers.
2. `optimal_tf` is mostly reduced to strategies, config, and CLI entry points.
3. the remaining `optimal_tf` wrappers are intentionally small.
4. docs have an explicit note that `trading_app_lab` is the target root name.
5. test and CLI validation pass before and after the path rename.

## Execution Summary

1. rename the repository root directory from `DMN` to `trading_app_lab`
2. update docs that embed absolute local paths
3. update examples that instruct users to `cd` into the old root name
4. review `pyproject.toml` metadata and decide whether the distribution name
   should also change
5. re-run the test suite and core CLI smoke tests
