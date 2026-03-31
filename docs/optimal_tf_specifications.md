# `optimal_tf` Project Specifications

Last updated: 2026-03-30

## Purpose

This document is the living functional specification for `optimal_tf`.

It should be updated:
- at the end of each session that changes project scope or behavior,
- or whenever explicitly requested.

This file is different from:
- `optimal_tf_architecture.md`: why the system is designed the way it is,
- `optimal_tf_usage.md`: how to run the system,
- `optimal_tf_strategies.md`: what each strategy means and how current strategy variants differ,
- this file: what the system is required to do, what it currently does, and what is intentionally deferred.

## Reading Guide

To keep this specification testable, each area is described with three lenses:
- `Normative requirements`
  The intended functional contract. These are the behaviors that tests and future changes should preserve unless the specification is revised.
- `Current implementation`
  The behavior currently implemented in the codebase.
- `Known gaps and planned changes`
  Important missing pieces, approximations, or near-term extensions.

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

### Normative requirements

The system must accept:
- a universe of tickers defined in `market_tickers_data`,
- a historical price source,
- a TOML config file,
- optional CLI overrides.

### Current implementation

The current implementation uses:
- `market_tickers_data` for universe definitions,
- `yfinance` for price history,
- TOML config loading through `config_io.py`,
- CLI overrides for a subset of config fields.

### Known gaps and planned changes

Planned improvements:
- optional alternative market data sources,
- more explicit asset metadata and benchmark metadata.

## Data Quality Requirements

### Normative requirements

The system must support a basic return-sanity filter:
- returns whose absolute value exceeds a configurable threshold must be excluded from estimation inputs,
- the same filter must be applied consistently to periodic evaluation,
- the same filter must be applied consistently to benchmark construction for charts.

The specification of "excluded" is:
- the anomalous asset return observation is treated as unavailable for that date,
- downstream components may ignore it or fill it only where their own aggregation rule requires a missing value policy,
- the policy must be applied consistently and documented.

Current default:
- `max_abs_return = 1.0`

### Current implementation

The implementation sanitizes returns before:
- covariance estimation,
- periodic evaluation,
- equal-weight benchmark construction.

When portfolio daily returns are aggregated, missing asset returns in the holding period are currently filled with `0.0` before cross-sectional aggregation.

### Known gaps and planned changes

Planned improvements:
- export diagnostics listing filtered anomalies by date and ticker,
- make the anomaly-handling policy more explicit in exported metadata.

## Portfolio Construction Requirements

### Normative requirements

The package must be able to:
- estimate rolling risk inputs from historical prices,
- build a covariance matrix from volatility and cleaned correlation estimates,
- compute portfolio weights using a named strategy,
- expose the list of supported strategy names through the CLI surface.

Every strategy must satisfy this contract:
- input: historical prices and estimation settings,
- output: an effective weight vector indexed by asset,
- strategies may additionally expose structured state such as `base_weights` or `signal_scale`,
- the output may be long-only or long/short depending on the requested convention,
- missing assets must resolve to `0.0` weight in exported panels,
- strategy naming must remain stable once exposed in config or CLI.

### Current implementation

Current strategy names exposed by the system are documented in:
- [optimal_tf_strategies.md](/Users/damien.figarol/DMN/docs/optimal_tf_strategies.md)

### Known gaps and planned changes

Strategy-specific evolution is tracked in:
- [optimal_tf_strategies.md](/Users/damien.figarol/DMN/docs/optimal_tf_strategies.md)

## Single-Date Allocation Requirements

### Normative requirements

The package must support:
- selecting an allocation date explicitly,
- defaulting to today when no date is given,
- resolving the effective market date as the latest available trading date on or before the requested date,
- computing and exporting the portfolio weights for that effective date.

If there is no eligible market date on or before the requested date, the system must fail explicitly.

### Current implementation

The current allocation flow:
- computes a full weight panel,
- resolves the allocation date on the panel index,
- returns the weight snapshot at the latest available market date on or before the requested date.

### Known gaps and planned changes

Planned improvements:
- make "today" resolution more explicit in exported metadata for reproducibility.

## Periodic Evaluation Requirements

### Normative requirements

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

The timing convention is:
- weights are observed at rebalance date `t`,
- those weights are applied only to returns strictly after `t`,
- the holding period ends at the next rebalance date if one exists,
- the next rebalance date is included in the previous holding period return stream,
- transaction cost for rebalance date `t` is charged on the first trading day strictly after `t`.

### Current implementation

The current engine:
- computes rebalance dates from the available market index,
- computes weights at each rebalance date,
- applies those weights to the following holding period,
- records gross and net daily returns,
- records holding-period gross and net returns,
- aggregates summary metrics over the evaluated window.

### Known gaps and planned changes

Planned improvements:
- richer end-to-end tests on realistic datasets,
- more explicit run metadata in exports,
- optional richer benchmark handling.

## Long-Only / Long-Short Requirements

### Normative requirements

The package must support two portfolio conventions:
- `long_only`
  Negative weights are clipped to zero, then renormalized.
- `long_short`
  Signed weights are preserved.

The chosen convention must apply consistently in single-date allocation and periodic evaluation.

### Current implementation

Current behavior:
- the long-only projection is applied after the base portfolio recipe,
- the same `long_only` flag is reused by allocation and evaluation flows.

### Known gaps and planned changes

No major gap identified here beyond stronger config validation and export metadata.

## Transaction Cost Requirements

### Normative requirements

At each rebalance:
- turnover is computed as the sum of absolute weight changes,
- transaction cost is computed as `cost_bps / 1e4 * turnover`,
- the cost is charged on the first day of the following holding period.

The current cost model is portfolio-level and deterministic.

### Current implementation

The current implementation:
- measures turnover as the `L1` difference between current and previous target weights,
- computes one scalar cost per rebalance,
- subtracts that cost from the first net return of the following holding period.

### Known gaps and planned changes

Planned improvements:
- slippage,
- asymmetric costs,
- asset-specific cost assumptions.

## Portfolio Volatility Targeting Requirements

### Normative requirements

The periodic evaluation engine must support portfolio-level volatility targeting:
- compute realized portfolio volatility from the gross return stream,
- use an EWMA estimator with configurable span,
- convert annual target volatility to daily target volatility,
- apply the scaling factor with a one-day lag to avoid look-ahead bias,
- clip the scale to a reasonable upper bound.

The targeting contract is return-level, not position-level:
- the scaled portfolio is defined through scaled return series,
- the exported rebalance weights are raw target weights unless otherwise specified by an export contract revision.

### Current implementation

Current implementation choices:
- target volatility source: gross portfolio return stream,
- scaling lag: one trading day,
- maximum scale: `5.0`.

### Known gaps and planned changes

Planned improvements:
- export the applied scale series explicitly,
- make volatility targeting more explicit in the exported portfolio state, not only in the return stream.

## Benchmark Requirements

### Normative requirements

The evaluation workflow must support comparison against simple universe-level references.

At minimum, the reporting layer must support:
- a universe equal-weight rebalanced reference,
- an equal-weight buy-and-hold reference.

Benchmark construction must use the same return-sanity filter as the evaluated portfolio.

### Current implementation

Current benchmarks:
- equal-weight rebalanced benchmark from sanitized daily returns,
- equal-weight buy-and-hold benchmark using fixed equal weights over the available asset universe.

### Known gaps and planned changes

Planned improvements:
- optional official universe benchmarks when available,
- more explicit benchmark metadata in outputs.

## Summary Metrics Requirements

### Normative requirements

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

Metric definitions must remain stable unless the specification is revised:
- `total_return`
  Cumulative compounded return over the evaluated net return series.
- `annualized_return`
  Annualized arithmetic mean of daily returns unless another definition is explicitly adopted later.
- `annualized_volatility`
  Daily return standard deviation times `sqrt(252)`.
- `sharpe_ratio`
  `annualized_return / annualized_volatility` with zero when annualized volatility is zero.
- `maximum_drawdown`
  Minimum drawdown of the compounded net asset value path.
- `average_turnover`
  Mean of the turnover series on the evaluation index used by the implementation.
- `annualized_turnover`
  `average_turnover * 252` under the current daily annualization convention.
- `total_transaction_cost`
  Sum of rebalance costs.
- `annualized_transaction_cost`
  `total_transaction_cost / evaluated_years`.
- `percentage_of_positive_days`
  Fraction of net return observations strictly greater than zero.
- `number_of_evaluated_days`
  Number of non-missing daily net return observations after evaluation-window slicing.
- `number_of_effective_rebalances`
  Number of rebalance dates that generate a non-empty holding period.

### Current implementation

The current code follows the definitions above and computes summary metrics from the net daily return stream.

### Known gaps and planned changes

Planned improvements:
- richer reporting:
  - drawdown charts,
  - rolling Sharpe,
  - rolling turnover,
  - exposure summaries,
  - risk contribution views.

## Export Requirements

### Normative requirements

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

Every export contract should specify:
- file name,
- index or primary key,
- column names,
- whether values are gross, net, raw, or targeted,
- enough metadata to reproduce the run.

### Current implementation

Current exported outputs include:
- single-date weights as CSV and JSON,
- `weights_by_rebalance.csv`,
- `base_weights_by_rebalance.csv`,
- `effective_weights_by_rebalance.csv`,
- `signal_scale.csv`,
- `portfolio_vol_scale.csv`,
- `daily_returns_gross.csv`,
- `daily_returns_net.csv`,
- `turnover.csv`,
- `costs.csv`,
- `summary.json`,
- `performance.png`.

### Known gaps and planned changes

Planned improvements:
- add run metadata to exports,
- add anomaly diagnostics export.

## Configuration Requirements

### Normative requirements

The TOML config currently includes:
- `[universe]`
- `[estimation]`
- `[backtest]`
- `[allocation]`
- `[evaluation]`

The config must remain:
- human-readable,
- suitable for command-line override,
- stable enough for scripted research runs.

The config surface must also define validation rules for:
- positive spans and window sizes,
- allowed strategy names,
- allowed rebalance frequencies,
- non-negative transaction costs,
- consistent date ranges.

### Current implementation

The current config loader maps TOML fields into typed dataclasses, but validation remains light and largely implicit.

### Known gaps and planned changes

Planned improvements:
- add explicit config validation with actionable error messages,
- stabilize the config surface once the strategy family is complete.

## Testing Requirements

### Normative requirements

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

Tests should also progressively cover:
- export contracts,
- benchmark definitions,
- configuration validation,
- end-to-end behavior on realistic datasets.

### Current implementation

The current suite already covers the main mechanics above.

### Known gaps and planned changes

Planned improvements:
- more end-to-end tests on realistic datasets,
- stronger tests around export schemas and diagnostics.

## Reproducibility Requirements

### Normative requirements

The system should make run conditions explicit enough for research reproduction:
- requested date vs effective allocation date,
- evaluation start and end dates,
- strategy and cleaning method,
- volatility targeting settings,
- transaction cost settings,
- universe identity.

### Current implementation

The CLI currently prints part of this information and includes a subset in exported files.

### Known gaps and planned changes

Planned improvements:
- include richer run metadata in export payloads,
- make date resolution and targeting choices more explicit.

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
