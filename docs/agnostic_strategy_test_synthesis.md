# `strategies_agnostic` Test Synthesis

Last updated: 2026-06-03

## Purpose

This note summarizes the exploratory tests run on the experimental
`src/optimal_tf/strategies_agnostic/` package, with a focus on:

- the Eq. 8 position engine,
- the `Q_phi = phi * C + (1 - phi) * I` family,
- the `ATF_AGNOSTIC` strategy,
- sensitivity to the trend horizon and to `phi`.

The goal is not to claim a universal optimum, but to record the main empirical
patterns observed so far.

## Experimental Reminder

Core Eq. 8 engine:

`w = omega * C^{-1/2} * Q^{-1/2} * p`

Current conventions used in the lab:

- `C` comes from the standard cleaned correlation pipeline.
- Structural `Q` models (`I`, `C`, `Q_phi`) receive only structural cleaning.
- `ATF_AGNOSTIC` uses `p = trend_ema(z_returns)`.
- `ARP_AGNOSTIC` and `PHI_*` with flat signal use `p = 1`.

Unless otherwise noted:

- evaluation window: `2010-01-01` to `2026-06-03`
- config: `configs/optimal_tf.example.toml`
- `phi=0.25` and `phi=0.75` are sometimes displayed by the scripts as
  `PHI_0.2` and `PHI_0.8` because of label rounding

Baseline test protocol:

- `C` cleaning method: `rie_reference`
- `covariance_window = 150`
- `covariance_min_periods = 60`
- `vol_span = 60`
- structural `Q` cleaning:
  - symmetrization
  - PSD projection
  - diagonal renormalization to `1`

For `ATF_AGNOSTIC`, the trend horizon is varied explicitly in the dedicated
tests; the covariance lookback remains fixed unless stated otherwise.

## Equity Markets

### `cac40`

Tests run:

- `compare_phi_grid` with `signal-family ones`
- `compare_phi_grid` with `signal-family trend_ema`
- `compare_atf_trend_sensitivity` with windows `21, 63, 126, 252` and
  `phi = 0.0, 0.5, 1.0`

Main findings:

- For `p = 1`, the best results were obtained with an intermediate `phi`,
  with a broad plateau around `0.4-0.6`.
- For `ATF_AGNOSTIC`, increasing `phi` generally hurt performance.
- Longer trend horizons were clearly better than short ones.

Representative results:

- `PHI_50` with `p = 1`: Sharpe `0.8565`
- `PHI_100` with `p = 1`: Sharpe `0.8266`
- `ARP_AGNOSTIC` with `p = 1`: Sharpe `0.7579`
- best `ATF_AGNOSTIC`: `EMA 252`, `phi=0.0`, Sharpe `0.7665`

Interpretation:

- On `cac40`, the agnostic TF signal looks too uncertain to justify a large
  move toward `Q = C`.
- For flat-signal Eq. 8 portfolios, however, an intermediate `phi` appears
  useful and robust.

### `eurostoxx50`

Test run:

- `compare_atf_trend_sensitivity` with windows `63, 126, 252, 378` and
  `phi = 0.0, 0.25, 0.5, 0.75, 1.0`

Main findings:

- The best `ATF_AGNOSTIC` variants use long trend windows.
- The best `phi` is clearly low, often exactly `0.0`.
- Raising `phi` steadily degrades performance.

Representative results:

- best case: `EMA 378`, `phi=0.0`, Sharpe `0.9558`
- `EMA 252`, `phi=0.0`: Sharpe `0.9487`
- `EMA 378`, `phi=1.0`: Sharpe `0.8554`

Interpretation:

- `eurostoxx50` behaves similarly to `cac40`.
- For these European equity universes, the TF signal benefits from long memory
  but remains better treated with a highly agnostic `Q`.

### `dji`

Test run:

- `compare_atf_trend_sensitivity` with windows `63, 126, 252, 378` and
  `phi = 0.0, 0.25, 0.5, 0.75, 1.0`

Main findings:

- Long trend windows again dominate short ones.
- Unlike the European equity universes, intermediate-to-high `phi` improves
  results.
- The best region is around `phi=0.5-0.75`.

Representative results:

- best case: `EMA 378`, `phi=0.75`, Sharpe `1.1791`
- `EMA 378`, `phi=0.5`: Sharpe `1.1749`
- `EMA 378`, `phi=0.0`: Sharpe `1.1598`
- `EMA 378`, `phi=1.0`: Sharpe `1.1766`

Interpretation:

- `dji` tells a more pro-signal story than the European markets.
- The gain over `phi=0` is not enormous, but it is persistent enough to make
  the intermediate/high `phi` region interesting.
- This result should still be read cautiously because `dji` is concentrated.

### `nasdaq100`

Test run:

- `compare_atf_trend_sensitivity` with windows `63, 126, 252, 378` and
  `phi = 0.0, 0.25, 0.5, 0.75, 1.0`

Main findings:

- Long trend windows strongly dominate short ones.
- Intermediate `phi` beats both `phi=0` and `phi=1`.
- The best region is around `phi=0.25`.

Representative results:

- best case: `EMA 252`, `phi=0.25`, Sharpe `1.3537`
- short windows were clearly weaker

Interpretation:

- `nasdaq100` reinforces the idea that US equity universes can support a less
  agnostic `Q` than the European universes tested so far.
- However, the optimum is not `phi=1`, so the data does not support complete
  confidence in the trend signal.

### `sp500`

Status:

- Several runs were started, including reduced grids, but did not complete in a
  practical amount of time during this session.
- No reliable market-level conclusion is recorded yet.

Interpretation:

- `sp500` should remain on the shortlist for the next session.
- It is especially important because it can tell us whether the `dji` and
  `nasdaq100` behavior generalizes to a broader US equity universe.

## Broad Index And Mixed Universes

### `world_index`

Test run:

- `compare_atf_trend_sensitivity` with windows `63, 126, 252, 378` and
  `phi = 0.0, 0.25, 0.5, 0.75, 1.0`

Main findings:

- Long trend windows clearly dominate.
- Intermediate/high `phi` improves the strategy.
- The best region is around `phi=0.75`, not at the extremes.

Representative results:

- best case: `EMA 378`, `phi=0.75`, Sharpe `0.8251`
- `EMA 378`, `phi=0.0`: Sharpe `0.7730`
- `EMA 378`, `phi=1.0`: Sharpe `0.7787`

Interpretation:

- `world_index` gives one of the clearest cases for a meaningful intermediate
  `phi`.
- This is one of the strongest arguments so far that `phi` is not just a
  numerical nuisance parameter.

### `dataset_all`

Test run:

- `compare_atf_trend_sensitivity` with windows `63, 126, 252, 378` and
  `phi = 0.0, 0.25, 0.5, 0.75, 1.0`

Main findings:

- Results are much weaker than on the equity-only universes.
- Short trend windows are poor and can be outright negative.
- Performance improves with longer windows and with larger `phi`.
- The best case is at `phi=1.0`.

Representative results:

- best case: `EMA 378`, `phi=1.0`, Sharpe `0.1537`
- `EMA 252`, `phi=1.0`: Sharpe `0.1461`
- many short-window cases are negative

Interpretation:

- `dataset_all` is probably too heterogeneous to support a clean ATF story.
- The weak absolute results matter more than the exact `phi` ranking.
- This universe is useful as a stress test, but not as a clean guide for
  choosing `phi`.

## Cross-Test Synthesis

### Stable observations

- Long trend windows consistently help `ATF_AGNOSTIC`.
- There is no evidence so far that short-horizon trend is the right way to use
  Eq. 8 in these universes.
- `ARP` is exactly recovered by `ARP_AGNOSTIC`.

### Less stable observations

- The best `phi` is not universal.
- European equity markets tested so far prefer low `phi`.
- US equity universes tested so far prefer intermediate or moderately high
  `phi`.
- `dataset_all` prefers `phi=1.0`, but this is not very informative because the
  strategy is weak there overall.

### Current practical reading

If the goal is to continue exploring `ATF_AGNOSTIC`, the most credible current
working hypothesis is:

- use long trend windows,
- do not assume a universal `phi`,
- analyze `phi` by market family,
- avoid drawing strong conclusions from overly heterogeneous universes.

## Suggested Next Steps

- Finish a usable `sp500` protocol, possibly with a lighter grid and explicit
  caching.
- Add plots of `Sharpe(phi)` by market for the long-window ATF variants.
- Compare the same market families with the flat-signal Eq. 8 family
  (`p = 1`) to see whether the geography effect comes from the signal or from
  the universe structure itself.

## Sensitivity To Covariance Window Scaling

A second ATF pass was run with:

`covariance_window = ceil(1.5 * N_assets)`

using the current asset count of each universe.

Scaled windows used:

- `cac40`: `60`
- `eurostoxx50`: `75`
- `dji`: `45`
- `nasdaq100`: `152`
- `world_index`: `26`
- `dataset_all`: `84`

Important implementation note:

- for `dji` and `world_index`, the scaled window is below the baseline
  `covariance_min_periods = 60`
- these two universes were rerun with
  `covariance_min_periods = covariance_window`
- without this adjustment, the protocol is not computable and collapses to
  zeroed strategies

### What remained stable

- `cac40` still prefers long trend windows and low `phi`, with the best case at
  `EMA 252`, `phi=0.0`, Sharpe `0.7630`
- `eurostoxx50` still prefers long trend windows and low `phi`, with the best
  case at `EMA 378`, `phi=0.0`, Sharpe `0.9570`
- `dji` still prefers long trend windows and intermediate/high `phi`, with the
  best case at `EMA 378`, `phi=0.75`, Sharpe `1.1800`
- `nasdaq100` still prefers long trend windows and intermediate `phi`, with the
  best case at `EMA 252`, `phi=0.25`, Sharpe `1.3550`

Interpretation:

- the main equity-market conclusions are robust to the shift from a fixed
  `covariance_window=150` to a universe-scaled window
- this is a stronger result than for the broad or heterogeneous universes

### What changed materially

- `world_index` no longer prefers `phi=0.75`; with `window=26` it moves back to
  a low-`phi` profile, best at `EMA 252`, `phi=0.0`, Sharpe `0.5422`
- `dataset_all` remains weak overall, but improves materially and still prefers
  high `phi`, best at `EMA 252`, `phi=1.0`, Sharpe `0.3278`

Interpretation:

- `world_index` is quite sensitive to the correlation lookback horizon
- `dataset_all` remains too heterogeneous to act as a stable calibration guide
- the most trustworthy cross-market ATF conclusions remain those from the
  cleaner equity universes

### Best-Sharpe Delta: `window=150` vs `window=1.5N`

Best Sharpe by universe:

- `cac40`: `0.7665 -> 0.7630` (`-0.0035`)
- `eurostoxx50`: `0.9558 -> 0.9570` (`+0.0012`)
- `dji`: `1.1791 -> 1.1800` (`+0.0009`)
- `nasdaq100`: `1.3537 -> 1.3550` (`+0.0013`)
- `world_index`: `0.8251 -> 0.5422` (`-0.2829`)
- `dataset_all`: `0.1537 -> 0.3278` (`+0.1741`)

Reading:

- `cac40`, `eurostoxx50`, `dji`, and `nasdaq100` are almost unchanged in best
  Sharpe terms.
- `world_index` is very sensitive to the covariance lookback choice.
- `dataset_all` improves with the shorter scaled window, but remains weak in
  absolute terms and should still be treated cautiously.
