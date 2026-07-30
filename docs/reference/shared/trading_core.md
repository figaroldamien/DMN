# `trading_core` Shared Package

Last updated: 2026-05-12

## Purpose

`trading_core` is the shared quant infrastructure layer for the repository.

It is meant to host the pieces that should be reusable across several trading
research applications, including:
- market universes and symbol normalization
- price loading
- return, volatility, and trend primitives
- covariance estimation and matrix cleaning
- rebalance calendars
- periodic backtest mechanics
- reporting, exports, and comparison outputs

`optimal_tf` is currently the main reference consumer of this package.

## Current Layout

`src/trading_core/`

- `market/`
  Shared access to market universes exposed by `market_tickers_data`.
- `data/`
  Shared price loading helpers and provider-specific symbol adaptation.
- `features/`
  Shared return, volatility, trend, and transformation primitives.
- `risk/`
  Shared covariance, correlation, cleaning, and estimation pipeline helpers.
- `rebalance/`
  Shared rebalance calendar utilities.
- `backtest/`
  Shared periodic evaluation engine and multi-strategy comparison helpers.
- `reporting/`
  Shared metrics, plots, benchmark builders, and export helpers.

## Public API

The top-level package now re-exports the most common primitives, so application
code can either:
- import from the focused submodules, or
- import from `trading_core` directly for high-level use cases.

Typical examples:

```python
from trading_core import load_prices_for_universe, estimate_clean_covariance_panel
from trading_core import evaluate_portfolio, compare_strategies
from trading_core import write_evaluation_outputs
```

For lower-level work, importing from submodules remains preferred when the call
site benefits from extra clarity.

## Boundary Rules

What belongs in `trading_core`:
- functions reusable by at least two trading apps
- infrastructure that is strategy-agnostic
- output contracts and reporting helpers that are not tied to one app

What should stay in an application package such as `optimal_tf`:
- strategy definitions
- app-specific config schemas
- app-specific CLI semantics
- research logic that is not yet stable enough to generalize

## Compatibility Policy

During the refactor, `optimal_tf` may keep thin compatibility wrappers around
the new core modules. Those wrappers are acceptable when they:
- preserve public imports or tests,
- keep the migration incremental,
- remain small and obvious.

The long-term goal is for application packages to become thinner, while
`trading_core` holds the shared mechanics.
