# ADR 0001: Introduce `trading_core`

Date: 2026-05-12

## Status

Accepted

## Context

The repository currently contains multiple trading research applications with
overlapping infrastructure:
- universe and ticker metadata
- market data loading
- return and volatility primitives
- covariance estimation and cleaning
- rebalance and evaluation mechanics
- reporting outputs

`optimal_tf` is the most active and complete application and should guide the
shared architecture.

## Decision

Introduce a new shared package:
- `src/trading_core`

The first extraction wave contains:
- `trading_core.market`
- `trading_core.data`
- `trading_core.features`
- `trading_core.rebalance`

`optimal_tf` becomes the first consumer of these shared modules.
`dmn` is adapted only for low-risk shared primitives.

## Consequences

Benefits:
- reduces duplicated quant infrastructure
- makes new research apps easier to create
- keeps strategy logic separate from shared mechanics

Trade-offs:
- temporary compatibility wrappers remain in `optimal_tf`
- `dmn` stays partly legacy for now

## Next Steps

Planned next extraction waves:
1. risk / covariance estimation: completed
2. backtest and evaluation engine: completed
3. reporting and export contracts: completed
4. comparison helpers and run outputs: completed
5. repository cleanup and rename preparation: in progress
