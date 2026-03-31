# `optimal_tf` Strategy Notes

Last updated: 2026-03-31

## Purpose

This document centralizes the description of the portfolio strategies available in `optimal_tf`.

It should be updated:
- when a strategy is added, renamed, or materially changed,
- or whenever explicitly requested.

See also:
- [optimal_tf_specifications.md](/Users/damien.figarol/DMN/docs/optimal_tf_specifications.md) for the functional contract that strategies must satisfy,
- [optimal_tf_architecture.md](/Users/damien.figarol/DMN/docs/optimal_tf_architecture.md) for design rationale and module layout,
- [optimal_tf_usage.md](/Users/damien.figarol/DMN/docs/optimal_tf_usage.md) for CLI and config usage.

## Strategy List

Current strategies:
- `RP`
- `ARP`
- `NM`
- `EW`
- `LLTF`
- `ToRP0`
- `ToRP1`
- `ToRP2`
- `ToRP3`

## Shared Conventions

All current strategies:
- produce one weight vector per date,
- consume historical prices through the existing date-centric estimation/allocation pipeline,
- can be projected to `long_only` or kept in `long_short`,
- are exposed through config and CLI by their strategy name.

The covariance-based strategies consume cleaned covariance estimates produced by the estimator pipeline.

## Strategy Descriptions

### `RP`

`RP` is the current risk-parity baseline.

Current implementation:
- uses inverse-volatility weights derived from the covariance diagonal,
- normalizes the resulting vector by gross exposure.

Practical interpretation:
- this is a simple and robust cross-asset baseline,
- it is currently implemented as an inverse-vol proxy, not as a full equal-risk-contribution solver.

## `ARP`

`ARP` is the agnostic risk parity strategy.

Current implementation:
- whitens the correlation structure,
- allocates so that decorrelated modes receive a more even ex-ante risk budget,
- maps the result back to asset space and normalizes by gross exposure.

Practical interpretation:
- `ARP` aims to balance risk across correlation eigenmodes rather than directly across assets,
- it is less tied to the dominant market mode than plain `RP`.

## `NM`

`NM` is the naive Markowitz strategy.

Current implementation:
- applies a pseudo-inverse to the covariance matrix,
- uses a flat expected-return vector,
- normalizes the resulting long/short allocation by gross exposure.

Practical interpretation:
- this is a covariance-driven allocation with minimal return assumptions,
- it is useful as a reference even though it can be unstable when covariance estimates are noisy.

## `EW`

`EW` is the equal-weight strategy.

Current implementation:
- allocates equally across currently available assets,
- recomputes the equal-weight vector on each date,
- applies the same long-only or long/short normalization path as the rest of the framework.

Practical interpretation:
- this is the simplest allocation baseline,
- it is useful as a sanity check and comparison anchor.

## `LLTF`

`LLTF` is an empirical lead-lag trend-following strategy inspired by Grebenkov and Serror (arXiv:1410.8409).

Current implementation:
- computes asset-level trend signals from volatility-normalized returns,
- builds virtual cross-asset return streams of the form `r_j * s_k`,
- estimates EWMA means and covariances on those virtual lead-lag streams,
- solves an empirical mean-variance problem for the lead-lag weight matrix,
- maps that weight matrix back into asset weights and normalizes the resulting vector.

Practical interpretation:
- `LLTF` lets each asset exposure react not only to its own signal but also to the signals of other assets,
- it is the first cross-asset TF strategy in the project whose core object is a signal-mixing matrix rather than a single per-asset overlay,
- this v1 is aligned with the article's intuition, but it remains an empirical implementation rather than a full reproduction of the paper's latent Gaussian model.

## `ToRP0`

`ToRP0` is the original trend-on-risk-parity implementation kept for comparison.

Current implementation:
- computes the `RP` portfolio as a base risk budget,
- estimates an asset-level trend signal from volatility-normalized returns,
- multiplies `RP` weights asset by asset by the corresponding asset trend signal,
- normalizes the resulting vector.

Practical interpretation:
- `ToRP0` is an overlay of individual asset trend signals on top of the `RP` budget,
- it is operational and intuitive,
- it is not the closest implementation of Sec. 3.4 of the reference paper.

## `ToRP1`

`ToRP1` is the current refinement of trend-on-risk-parity.

Current implementation:
- computes the `RP` portfolio,
- estimates an asset-level trend signal from volatility-normalized returns,
- projects that signal onto the `RP` factor,
- applies the resulting common factor signal back onto the `RP` portfolio,
- normalizes the resulting vector.

Practical interpretation:
- `ToRP1` measures one common trend on the `RP` factor instead of keeping separate asset-level overlays,
- this is closer to Sec. 3.4 of the reference paper than `ToRP0`,
- it remains a simplified implementation because the framework still renormalizes weights each period and does not yet preserve the full signal amplitude as a separate leverage layer.

## `ToRP2`

`ToRP2` is the current article-aligned `ToRP` implementation.

Current implementation:
- builds an `RP` factor with category-aware tilt when metadata is available,
- sets the `RP` tilt to zero on FX instruments, following the paper's convention for the special `RP` direction,
- computes the trend signal on the `RP` factor return stream itself rather than projecting separate asset signals afterward,
- applies that common factor signal back onto the tilted `RP` portfolio,
- normalizes the resulting vector within the current framework.
- in periodic evaluation, the historical `RP` factor path is now cached once per run and reused across rebalance dates.

Practical interpretation:
- `ToRP2` is the most paper-aligned `ToRP` variant currently implemented in the project,
- it differs from `ToRP1` by using the trend of the `RP` portfolio itself as the signal source,
- it still inherits one framework simplification: weights are normalized in the allocation layer rather than exported with a separate leverage amplitude.

## `ToRP3`

`ToRP3` is the first `ToRP` variant that keeps the amplitude of the factor signal explicit in the strategy state.

Current implementation:
- builds a category-aware `RP` base portfolio, with FX neutralized when metadata is available,
- computes the trend signal on the `RP` factor return stream itself,
- normalizes that factor trend by the factor's own EWMA volatility,
- applies `torp_signal_gain` to the normalized factor signal,
- stores that scaled, volatility-normalized signal as `signal_scale`,
- applies that signal to the `RP` base portfolio without renormalizing away the amplitude,
- exposes both `base_weights` and `effective_weights`.
- in periodic evaluation, the `RP` factor base path and factor signal context are cached once per run and reused across rebalance dates.

Practical interpretation:
- `ToRP3` keeps the difference between weak and strong trend conviction,
- it is the closest current strategy to the article once signal amplitude matters,
- it is the natural base for combining `ToRP` signal conviction with portfolio-level volatility targeting.

## Current Gaps

Important current gaps:
- `RP` is still an inverse-volatility proxy rather than a full ERC implementation,
- `ToRP3` preserves signal amplitude, but the framework still layers a separate portfolio volatility target on top rather than exporting a single unified leverage decision,
- there is no final generalized optimal portfolio built as a calibrated combination of the basic blocks,
- strategy metadata is documented here but not yet exported systematically with run outputs.

## Planned Strategy Work

Planned improvements:
- refine `ToRP` toward a more paper-faithful implementation,
- explore portfolio combinations inspired by the note:
  - `ARP + ToRP`
  - `ARP + RP + ToRP`
- compare strategy behavior under:
  - `empirical`
  - `linear_shrinkage`
  - `rie`
