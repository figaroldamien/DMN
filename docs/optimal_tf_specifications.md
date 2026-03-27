# `optimal_tf` Project Specifications

Last updated: 2026-03-27

## Purpose

This document is the living functional specification for `optimal_tf`.

It should be updated:
- at the end of each session that changes project scope or behavior,
- or whenever explicitly requested.

This file is different from:
- `optimal_tf_architecture.md`: why the system is designed the way it is,
- `optimal_tf_usage.md`: how to run the system,
- this file: what the system is supposed to do.

## Project Goal

`optimal_tf` is a research and implementation package for cross-asset portfolio construction inspired by the paper:
- `2201.06635v1` "Optimal trend following portfolios"

The project aims to:
- estimate cross-asset risk structure from market data,
- construct portfolio weights from analytical recipes,
- evaluate those portfolios over time under configurable rebalance schedules,
- compare implementations and later validate a native RIE cleaner against an external reference.

## Current Functional Scope

The current scope includes:
- loading price data from a universe of tickers,
- computing portfolio weights at a given date,
- running periodic evaluations with scheduled rebalancing,
- supporting long-only and long/short allocations,
- applying transaction costs,
- applying portfolio-level volatility targeting,
- exporting results,
- reporting summary metrics.

## Inputs

The system currently expects:
- a universe of tickers defined in `market_tickers_data`,
- a price history source, currently `yfinance`,
- a TOML config file,
- optional CLI overrides.

## Data Quality Requirements

The system must support a basic return-sanity filter:
- returns whose absolute value exceeds a configurable threshold must be excluded,
- the same filter must be applied consistently to:
  - estimation inputs,
  - periodic evaluation,
  - benchmark construction for charts.

Current default:
- `max_abs_return = 1.0`

## Portfolio Construction Requirements

The package must be able to:
- estimate rolling risk inputs from historical prices,
- build a covariance matrix from volatility and cleaned correlation estimates,
- compute portfolio weights using a named strategy.

Current strategies:
- `RP`
- `ARP`
- `NM`
- `EW`
- `ToRP`

Planned strategies:
- paper-faithful refined variants of `ToRP`

## Single-Date Allocation Requirements

The package must support:
- selecting an allocation date explicitly,
- defaulting to today when no date is given,
- resolving the effective market date as the latest available trading date on or before the requested date,
- computing and exporting the portfolio weights for that date.

## Periodic Evaluation Requirements

The package must support:
- discrete rebalancing at a configurable frequency,
- recalculating weights at each rebalance date,
- applying those weights on the following holding period,
- computing gross and net returns,
- accounting for turnover and transaction costs,
- aggregating performance over the evaluation window.

Supported rebalance frequencies:
- `daily`
- `weekly`
- `monthly`
- `quarterly`
- `yearly`

## Long-Only / Long-Short Requirements

The package must support two portfolio conventions:
- `long_only`
  Negative weights are clipped to zero, then renormalized.
- `long_short`
  Signed weights are preserved.

## Transaction Cost Requirements

At each rebalance:
- turnover is computed as the sum of absolute weight changes,
- transaction cost is computed as `cost_bps / 1e4 * turnover`,
- the cost is charged on the first day of the following holding period.

## Portfolio Volatility Targeting Requirements

The periodic evaluation engine must support portfolio-level volatility targeting:
- compute realized portfolio volatility from the gross return stream,
- use an EWMA estimator with configurable span,
- convert annual target volatility to daily target volatility,
- apply the scaling factor with a one-day lag to avoid look-ahead bias,
- clip the scale to a reasonable upper bound.

Current implementation choices:
- target volatility source: gross portfolio return stream,
- scaling lag: one trading day,
- maximum scale: `5.0`.

## Summary Metrics Requirements

The periodic evaluation engine must report:
- total return,
- annualized return,
- annualized volatility,
- Sharpe ratio,
- maximum drawdown,
- average turnover,
- annualized turnover,
- total transaction cost,
- annualized transaction cost,
- percentage of positive days,
- number of evaluated days,
- number of effective rebalances.

## Export Requirements

The system must support exporting:
- weights at one date,
- weights by rebalance date,
- daily gross returns,
- daily net returns,
- turnover by rebalance,
- cost by rebalance,
- summary metrics,
- a chart comparing:
  - the `optimal_tf` portfolio,
  - a universe equal-weight reference index,
  - an equal-weight buy-and-hold benchmark.

## Configuration Requirements

The TOML config currently includes:
- `[universe]`
- `[estimation]`
- `[backtest]`
- `[allocation]`
- `[evaluation]`

The config must remain human-readable and suitable for command-line override.

## Testing Requirements

The project must keep automated tests for:
- matrix conversion and PSD repair,
- cleaner behavior,
- portfolio weight normalization,
- allocation date resolution,
- rebalance date generation,
- periodic evaluation mechanics,
- transaction cost application,
- volatility targeting behavior,
- CLI behavior and exports.

## Current Non-Goals

Not yet required in the current version:
- a production-grade execution engine,
- futures contract roll logic,
- slippage by asset class,
- broker integration,
- a paper-faithful refined `ToRP`.

## Remaining Work And Priorities

### Priority 1

- Validate the native `RIE` cleaner against an external reference implementation.
- Add a diagnostics export listing filtered return anomalies by date and ticker.
- Refine `ToRP` toward a more paper-faithful implementation.

Current planned validation sequence for `RIE`:
1. add a comparison tool against an external implementation or trusted reference output,
2. run comparative backtests with:
   - `empirical`
   - `linear_shrinkage`
   - `rie`
3. measure the impact of the cleaning choice on:
   - `ARP`
   - `NM`
   - `ToRP`

### Priority 2

- Add portfolio combinations inspired by the note:
  - `ARP + ToRP`
  - `ARP + RP + ToRP`
- Improve benchmark selection with optional official universe benchmarks when available.
- Make volatility targeting more explicit in the exported portfolio state, not only in the return stream.

### Priority 3

- Add richer reporting:
  - drawdown charts,
  - rolling Sharpe,
  - rolling turnover,
  - exposure summaries,
  - risk contribution views.
- Add richer transaction cost models:
  - slippage,
  - asymmetric costs,
  - asset-specific cost assumptions.

### Priority 4

- Improve data robustness:
  - better handling of broken Yahoo series,
  - optional alternative market data sources,
  - more explicit asset metadata and benchmark metadata.

### Priority 5

- Add more end-to-end tests on realistic datasets.
- Stabilize the config surface once the strategy family is complete.
