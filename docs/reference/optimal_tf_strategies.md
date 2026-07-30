# `optimal_tf` Strategy Notes

Last updated: 2026-06-11

## Purpose

This document centralizes the description of the portfolio strategies available
in `optimal_tf`.

It should be updated:
- when a strategy is added, renamed, or materially changed,
- or whenever explicitly requested.

Workflow note:
- material strategy evolutions should also be logged in
  `docs/research/strategy_evolution_log.md`,
- durable methodological or architectural choices should be captured in
  `docs/adr/`.

See also:
- [optimal_tf_specifications.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_specifications.md) for the functional contract that strategies must satisfy,
- [optimal_tf_architecture.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_architecture.md) for design rationale and module layout,
- [optimal_tf_usage.md](/Users/damien.figarol/trading_app_lab/docs/user_guides/optimal_tf_usage.md) for CLI and config usage,
- [strategy_evolution_log.md](/Users/damien.figarol/trading_app_lab/docs/research/strategy_evolution_log.md) for the running change log of strategy evolutions and deferred ideas,
- [agnostic_strategy_test_synthesis.md](/Users/damien.figarol/trading_app_lab/docs/research/agnostic_strategy_test_synthesis.md) for the running empirical test log of the Eq. 8 lab,
- [agnostic_arp_phi_synthesis.md](/Users/damien.figarol/trading_app_lab/docs/research/agnostic_arp_phi_synthesis.md) for the dedicated `ARP_AGNOSTIC` `phi` sensitivity study.

Implementation note:
- production and baseline strategy code now lives primarily under
  `src/optimal_tf/strategies/`,
- the Eq. 8 research lab lives under `src/optimal_tf/strategies_agnostic/`,
- shared estimation, backtest, and reporting mechanics now live mostly under
  `src/trading_core/`.

## Strategy Surface

### Main documented strategies

Current main strategy surface:
- `RP`
- `ARP`
- `NM`
- `EW`
- `LLTF`

These are the strategies that should currently be considered the active core of
the project.

### Agnostic research recipes

Current Eq. 8 research recipes:
- `ARP_AGNOSTIC`
- `MARKOWITZ_AGNOSTIC`
- `ATF_AGNOSTIC`
- `ATF_RAW`
- `ATF_EMPIRICAL_Q`
- `PHI_0`
- `PHI_25`
- `PHI_50`
- `PHI_100`

These belong to the research lab, not yet to the main user-facing strategy
surface.

### Legacy note

Older `ToRP` variants are intentionally omitted from this document.

Reason:
- they are no longer the preferred direction of strategy development,
- the current project focus has moved toward the covariance strategies and the
  Eq. 8 agnostic research family,
- keeping `ToRP` in the main strategy notes now adds more historical noise than
  practical value.

If needed later, `ToRP` should be documented in a dedicated archival note rather
than in the main strategy overview.

## Shared Conventions

All current strategies:
- produce one weight vector per date,
- consume historical prices through the date-centric estimation/allocation
  pipeline,
- can be projected to `long_only` or kept in `long_short`,
- are addressed by explicit strategy or recipe names in the current config and
  CLI surfaces, even though not all of them should be considered equally
  stable.

The covariance-based strategies consume cleaned covariance estimates produced by
the standard estimator pipeline.

Current estimator convention:
- the standard covariance path uses a fixed historical window plus matrix
  cleaning,
- `EWMA` remains in use for volatility normalization, trend signals, and
  portfolio volatility targeting,
- `LLTF` is a special case because it still estimates EWMA moments internally
  on its virtual lead-lag streams.

Execution note:
- in periodic evaluation, the shared `trading_core.backtest` engine handles
  rebalance scheduling, holding-period rollout, turnover, costs, and reporting
  exports,
- strategy modules mainly provide the portfolio state computed at each rebalance
  date,
- implemented weights may now differ from raw target weights because the shared
  portfolio layer can apply weight smoothing.

## General Principles

### Main difference between strategies

The main difference between the strategies is the way they compute portfolio
positions.

Common backbone:
- the same price history is loaded,
- the same rebalance schedule is applied,
- the same backtest engine computes turnover, costs, and realized returns,
- the same reporting layer computes summary metrics and exports.

What changes from one strategy to another is mainly the position engine:
- how the raw signal vector is defined,
- whether the strategy uses no matrix, a covariance matrix, or a correlation
  matrix,
- how that matrix is inverted, whitened, or otherwise transformed into
  positions.

### Eq. 8

The agnostic research family is organized around the Eq. 8 position engine:

`w = omega * C^{-1/2} * Q^{-1/2} * p`

where:
- `w` is the portfolio weight vector,
- `omega` is a scalar used to set the overall portfolio scale,
- `C` is the cleaned empirical correlation matrix,
- `Q` is the signal correlation matrix, or more generally a structural matrix normalized as a correlation matrix,
- `p` is the signal vector.

Interpretation:
- `p` tells the strategy what directional view to express,
- `Q` describes how the signal co-moves across assets before mapping it to
  portfolio weights,
- `C^{-1/2}` whitens the correlation structure so that crowded correlation
  modes do not dominate mechanically,
- `omega` rescales the final vector to the desired exposure convention.

This Eq. 8 family is useful because many recipes can be understood as changes
to only two ingredients:
- the signal model `p`,
- the signal correlation matrix `Q`.

### Weight smoothing

After a strategy computes raw target weights, the portfolio layer may smooth
them before implementation.

Current smoothing rule:

`w_t^{impl} = alpha * w_t^{target} + (1 - alpha) * w_{t-1}^{impl}`

where:
- `w_t^{target}` is the newly computed target weight vector,
- `w_t^{impl}` is the implemented weight vector after smoothing,
- `w_{t-1}^{impl}` is the previously implemented vector,
- `alpha` is the weight smoothing coefficient.

Interpretation:
- `alpha = 1` means no smoothing,
- smaller `alpha` means slower portfolio adjustment,
- smoothing usually reduces turnover and trading costs,
- but it also delays the reaction to new estimates and new signals.

### General workflow

At a high level, the workflow is:

1. Load prices and define the evaluation dates.
2. Build the estimator inputs from the historical window available at each
   rebalance date.
3. Estimate the relevant objects used by the strategy:
   covariance, correlation, trend signals, or lead-lag objects.
4. Compute raw target positions from the selected strategy.
5. Optionally smooth the weights before implementation.
6. Roll the portfolio forward over the next holding period.
7. Compute turnover, costs, gross returns, net returns, and summary metrics.

This is why most strategy comparisons in `optimal_tf` should be read as
comparisons of the position engine under a mostly shared execution and
reporting stack.

## Main Strategy Descriptions

## Strategy Taxonomy

### Strategies independent from full covariance/correlation matrices

These strategies do not use the full covariance or correlation matrix as their
main position engine:

- `EW`: equal-weight baseline with no matrix input,
- `RP`: inverse-volatility baseline driven only by per-asset volatility, not by
  the full covariance or correlation structure.

### Strategies using covariance matrices

These strategies use the covariance matrix directly in the position
construction:

- `NM`: pseudo-inverse covariance allocation with a flat expected-return
  vector,
- `LLTF`: empirical mean-variance allocation on virtual lead-lag return
  streams, using internally estimated EWMA covariance objects.

### Strategies using correlation matrices

These strategies use the correlation matrix explicitly in the position engine:

- `ARP`: diagonalizes and whitens the cleaned correlation matrix,
- `ARP_AGNOSTIC`: Eq. 8 recipe using cleaned correlation plus `Q = I`,
- `MARKOWITZ_AGNOSTIC`: Eq. 8 recipe using cleaned correlation plus `Q = C`,
- `ATF_AGNOSTIC`: Eq. 8 trend-following recipe using cleaned correlation plus
  `Q = I`,
- `ATF_RAW`: same signal family as `ATF_AGNOSTIC`, but without gross
  normalization of final weights,
- `ATF_EMPIRICAL_Q`: Eq. 8 trend-following recipe using cleaned correlation
  plus an empirical `Q`,
- `PHI_0`, `PHI_25`, `PHI_50`, `PHI_100`: Eq. 8 recipes using cleaned
  correlation plus a `Q_phi` interpolation.

### `RP`

`RP` is the current risk-parity baseline.

Current implementation:
- uses inverse-volatility weights derived from the covariance diagonal,
- normalizes the resulting vector by gross exposure,
- is intentionally a robust proxy rather than a full equal-risk-contribution
  optimizer.

Practical interpretation:
- this is the simplest risk-budgeting baseline in the project,
- it is mostly driven by per-asset volatility scale,
- it does not explicitly whiten or redistribute correlation modes.

### `ARP`

`ARP` is the current agnostic risk parity strategy.

Current implementation:
- converts covariance to correlation,
- diagonalizes correlation into orthogonal modes,
- whitens those modes by dividing by the square root of the eigenvalues,
- maps the result back into asset space,
- normalizes by gross exposure.

Practical interpretation:
- `ARP` balances exposure across decorrelated correlation modes rather than
  directly across assets,
- it is the main non-trivial covariance strategy in the current baseline
  surface,
- it is the closest historical strategy to the Eq. 8 agnostic lab.

### `NM`

`NM` is the naive Markowitz strategy.

Current implementation:
- applies a pseudo-inverse to the covariance matrix,
- uses a flat expected-return vector,
- normalizes the resulting long/short allocation by gross exposure.

Practical interpretation:
- this is the simplest covariance-inverse portfolio in the project,
- it is useful as a diagnostic strategy because it reveals what the covariance
  estimate alone implies when every asset is assigned the same expected return,
- it can be more unstable than `RP` or `ARP` when the covariance estimate is
  noisy.

### `EW`

`EW` is the equal-weight strategy.

Current implementation:
- allocates equally across currently available assets,
- recomputes the equal-weight vector on each date,
- applies the same normalization conventions as the rest of the framework.

Practical interpretation:
- this is the simplest allocation baseline,
- it is mainly a sanity-check and benchmark anchor,
- it is not covariance-driven.

### `LLTF`

`LLTF` is an empirical lead-lag trend-following strategy inspired by Grebenkov
and Serror.

Current implementation:
- computes asset-level trend signals from volatility-normalized returns,
- builds virtual cross-asset return streams of the form `r_j * s_k`,
- estimates EWMA means and covariances on those virtual lead-lag streams,
- solves an empirical mean-variance problem for the lead-lag weight matrix,
- maps that matrix back into asset weights and normalizes the resulting vector.

Practical interpretation:
- `LLTF` is the current main cross-asset trend-following strategy,
- its core object is a signal-mixing matrix rather than a direct Eq. 8 closed
  form,
- it is conceptually separate from both the base covariance strategies and the
  agnostic lab.

## Agnostic Lab

The `strategies_agnostic` package is a research sandbox for strategies built
around the Eq. 8 position engine:

`w = omega * C^{-1/2} * Q^{-1/2} * p`

where:
- `C` is the cleaned empirical correlation matrix,
- `Q` is a second matrix controlling the structural recipe,
- `p` is the signal vector,
- `omega` is a scalar amplitude.

Current design decisions:
- `C` comes from the standard cleaned-correlation estimator path,
- structural `Q` models such as `I`, `C`, and `Q_phi` are not re-cleaned with
  the user-selected statistical shrinkage method,
- those structural `Q` matrices only receive structural repair:
  symmetrization, PSD projection, and diagonal renormalization,
- empirical `Q` models built from observed signal history may use the same
  statistical cleaners as `C`.

Why:
- `rie_reference` naturally applies to empirical sample matrices,
- structural `Q` models are analytical constructions,
- statistically shrinking them again would blur the meaning of the closed-form
  recipes.

### Signal models

Current agnostic signal models:
- `ones`
- `trend_ema`

Interpretation:
- `ones` means the strategy is purely structural and does not use a directional
  trend signal,
- `trend_ema` introduces an asset-level trend vector built from
  volatility-normalized returns.

### `Q` models

Current agnostic `Q` models:
- `identity`
- `correlation`
- `phi_shrink_correlation`
- `empirical`

Interpretation:
- `identity` gives the most agnostic structural benchmark,
- `correlation` pushes the recipe toward a correlation-inverse / Markowitz-like
  structure,
- `phi_shrink_correlation` interpolates between the two,
- `empirical` builds `Q` from signal-history dependence rather than from a
  purely structural formula.

## Agnostic Recipe Descriptions

### `ARP_AGNOSTIC`

Recipe:
- `signal_model = ones`
- `q_model = identity`

Interpretation:
- this is the cleanest Eq. 8 analogue of the current `ARP` intuition,
- it produces a mode-balanced structural allocation without any directional
  signal.

### `MARKOWITZ_AGNOSTIC`

Recipe:
- `signal_model = ones`
- `q_model = correlation`

Interpretation:
- this is the Eq. 8 recipe that sits closest to a covariance-inverse /
  Markowitz-style logic,
- it is the agnostic lab counterpart of `NM`, although not an exact replica of
  the historical `NM` implementation.

### `ATF_AGNOSTIC`

Recipe:
- `signal_model = trend_ema`
- `q_model = identity`
- `normalization = gross`

Interpretation:
- this is the trend-signaled extension of `ARP_AGNOSTIC`,
- it keeps the Eq. 8 structural `Q = I` logic while replacing the flat signal
  `p = 1` with a trend signal.

### `ATF_RAW`

Recipe:
- `signal_model = trend_ema`
- `q_model = identity`
- `normalization = raw`

Interpretation:
- same structural recipe as `ATF_AGNOSTIC`,
- but keeps the raw Eq. 8 amplitude before the framework’s gross normalization.

### `ATF_EMPIRICAL_Q`

Recipe:
- `signal_model = trend_ema`
- `q_model = empirical`

Interpretation:
- this extends the agnostic lab by making both the signal and the `Q` matrix
  data-driven,
- it is more exploratory and less structurally interpretable than the
  `identity` and `phi` recipes.

### `PHI_0`, `PHI_25`, `PHI_50`, `PHI_100`

Recipe family:
- `signal_model = ones`
- `q_model = phi_shrink_correlation`
- `phi in {0.0, 0.25, 0.5, 1.0}`

Interpretation:
- `PHI_0` is equivalent in spirit to `ARP_AGNOSTIC`,
- `PHI_100` is equivalent in spirit to `MARKOWITZ_AGNOSTIC`,
- intermediate `phi` values define a continuous bridge between these two poles.

This `phi` family is currently one of the main research directions in the
agnostic lab.

## Link Between Main Strategies And Agnostic Recipes

The agnostic lab is not a separate product line. It is a way to restate and
generalize part of the existing strategy surface inside one closed-form Eq. 8
engine.

### Best conceptual correspondences

- `ARP` <-> `ARP_AGNOSTIC`
  Both aim to balance exposure across decorrelated modes rather than simply
  using diagonal volatility scaling.

- `NM` <-> `MARKOWITZ_AGNOSTIC`
  Both are the “inverse structured risk matrix times flat signal” member of
  their respective families.

- `ATF_AGNOSTIC` <-> “trend-signaled extension of ARP-style logic”
  This is not the same object as `LLTF`, but it is the most direct Eq. 8 way to
  add time-series trend information on top of the agnostic structural engine.

- `PHI_*` <-> bridge between `ARP`-like and `NM`-like logic
  This family is useful precisely because it expresses the transition between
  the two poles in one parametric continuum.

### Important non-equivalences

- `RP` has no exact agnostic counterpart in the current lab.
  `RP` is a diagonal inverse-volatility rule, while the agnostic engine is built
  around `C^{-1/2} Q^{-1/2} p`.

- `EW` has no agnostic counterpart.
  It is a pure equal-allocation baseline rather than a matrix-driven strategy.

- `LLTF` has no direct Eq. 8 counterpart.
  It is based on cross-asset lead-lag signal mixing, not on a single closed-form
  Eq. 8 position engine.

### Practical reading

The simplest way to read the current roadmap is:
- keep `RP`, `ARP`, `NM`, `EW`, and `LLTF` as the main operational baselines,
- use the agnostic lab to understand, simplify, and possibly generalize the
  covariance-based part of the strategy stack,
- treat `ARP_AGNOSTIC`, `MARKOWITZ_AGNOSTIC`, and the `PHI_*` family as the key
  bridge between the historical strategies and the Eq. 8 research program.

## Current Gaps

Important current gaps:
- `RP` is still an inverse-volatility proxy rather than a full ERC
  implementation,
- the agnostic lab is richer conceptually than the main strategy surface, but
  it is still a research workflow rather than a stabilized production interface,
- there is no final generalized portfolio that cleanly unifies the best
  covariance baseline, trend overlay, and implementation-cost layer,
- strategy metadata is documented here but not yet exported systematically with
  run outputs.

## Planned Strategy Work

Planned improvements:
- keep developing the Eq. 8 agnostic family and its `phi` interpolation logic,
- test whether some agnostic recipes should graduate into the main documented
  strategy surface,
- continue studying turnover-aware implementation overlays on top of the raw
  target portfolios,
- preserve historical strategies only when they still add genuine explanatory
  value to the research stack.
