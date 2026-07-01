# `ARP_AGNOSTIC` Phi Sensitivity Synthesis

Last updated: 2026-06-03

## Purpose

This note summarizes the `phi` sensitivity study run on the flat-signal Eq. 8
family, i.e. the `ARP_AGNOSTIC`-style strategies built with:

`w = omega * C^{-1/2} * Q_phi^{-1/2} * 1`

where:

`Q_phi = phi * C + (1 - phi) * I`

The objective is to understand whether a stable preference for intermediate
`phi` emerges across markets, or whether the best `phi` remains market-specific.

## Experimental Setup

Strategy family:

- `signal-family ones`
- `q_model = phi_shrink_correlation`
- `normalization = gross`

Command family used:

```bash
.venv/bin/python -m optimal_tf.scripts.evaluation.compare_phi_grid \
  --config configs/optimal_tf.example.toml \
  --universe <universe> \
  --signal-family ones \
  --phi-step 0.1 \
  --evaluation-start 2010-01-01 \
  --evaluation-end 2026-06-03
```

Conventions:

- `phi` is explored on `0.0, 0.1, ..., 1.0`
- all results use the current cleaned-correlation pipeline for `C`
- structural `Q_phi` matrices receive structural cleaning only

Baseline test protocol used in this first study:

- `C` cleaning method: `rie_reference`
- `covariance_window = 150`
- `covariance_min_periods = 60`
- `vol_span = 60`
- `Q_phi` cleaning: explicit structural repair only
  - symmetrization
  - PSD projection
  - diagonal renormalization to `1`

## Equity Markets

### `cac40`

Main pattern:

- Clear hump-shaped profile.
- Best region is a broad plateau around `phi=0.4-0.6`.
- `phi=0.5` is the best point in the tested grid.

Representative results:

- `phi=0.0`: Sharpe `0.7579`
- `phi=0.4`: Sharpe `0.8564`
- `phi=0.5`: Sharpe `0.8565`
- `phi=1.0`: Sharpe `0.8266`

Reading:

- `cac40` is one of the strongest cases for a meaningful intermediate `phi`.
- The fact that the optimum is broad is more convincing than a very sharp peak.

### `eurostoxx50`

Main pattern:

- Same broad hump-shaped profile as `cac40`.
- Best region is again around `phi=0.4-0.6`.
- `phi=0.5` is the best grid point.

Representative results:

- `phi=0.0`: Sharpe `0.9317`
- `phi=0.4`: Sharpe `1.0751`
- `phi=0.5`: Sharpe `1.0777`
- `phi=1.0`: Sharpe `1.0354`

Reading:

- `eurostoxx50` strongly confirms the `cac40` pattern.
- For these European equity universes, `ARP_AGNOSTIC` seems to benefit from a
  genuine compromise between full agnosticism and full confidence in `C`.

### `dji`

Main pattern:

- Performance improves almost monotonically with `phi`.
- The best point in the tested grid is `phi=1.0`.
- The gain is gradual, not abrupt.

Representative results:

- `phi=0.0`: Sharpe `1.2590`
- `phi=0.2`: Sharpe `1.2699`
- `phi=0.5`: Sharpe `1.2683`
- `phi=1.0`: Sharpe `1.2754`

Reading:

- `dji` is not a good case for a clean intermediate optimum.
- The market still shows a broad flat region, but the best point lies at the
  fully confident end of the grid.
- This should be read cautiously because `dji` is a narrow, concentrated index.

### `nasdaq100`

Main pattern:

- Clear intermediate optimum.
- Best region is around `phi=0.1-0.3`.
- `phi=0.2` is the best point in the tested grid.

Representative results:

- `phi=0.0`: Sharpe `1.4248`
- `phi=0.1`: Sharpe `1.4497`
- `phi=0.2`: Sharpe `1.4531`
- `phi=0.5`: Sharpe `1.4464`
- `phi=1.0`: Sharpe `1.3466`

Reading:

- `nasdaq100` does support an intermediate `phi`, but at a lower level than the
  European universes.
- It is also one of the clearest cases where `phi=1.0` is too aggressive.

## Broad Index And Mixed Universes

### `world_index`

Main pattern:

- Weak preference for low `phi`.
- `phi=0.0` is the best point in the tested grid.
- The profile is not sharply monotonic, but `phi=1.0` is clearly worse.

Representative results:

- `phi=0.0`: Sharpe `0.7046`
- `phi=0.5`: Sharpe `0.6770`
- `phi=0.9`: Sharpe `0.6862`
- `phi=1.0`: Sharpe `0.6672`

Reading:

- `world_index` does not support the same intermediate-`phi` story as European
  equities.
- The best interpretation is probably that the broad aggregation of countries
  reduces the value of moving strongly toward `Q = C`.

### `dataset_all`

Main pattern:

- Strongly decreasing profile as `phi` increases.
- `phi=0.0` is the best point in the tested grid.
- `phi=1.0` is materially worse.

Representative results:

- `phi=0.0`: Sharpe `0.5592`
- `phi=0.5`: Sharpe `0.5246`
- `phi=1.0`: Sharpe `0.4490`

Reading:

- `dataset_all` is the clearest case for staying close to the agnostic end.
- Because this universe is highly heterogeneous, the result is useful as a
  robustness warning rather than as a precise calibration guide.

## Cross-Market Synthesis

### What looks stable

- There is no universal `phi`.
- `phi=1.0` is rarely dominant in broad or heterogeneous universes.
- Intermediate `phi` is often helpful in equity-only universes.

### What looks especially convincing

- `cac40` and `eurostoxx50` both show a broad optimum around `phi=0.5`.
- `nasdaq100` shows a clear intermediate optimum, but lower, around `phi=0.2`.

These three markets make a reasonably strong case that intermediate `phi` is a
real empirical feature of the flat-signal Eq. 8 family, not just a numerical
artifact.

### What remains unstable

- `dji` prefers `phi=1.0`, although the surface is fairly flat once `phi` is
  moderately positive.
- `world_index` and `dataset_all` prefer low `phi`.

This means the interpretation should remain conditional on market family:

- European equity: `phi` around `0.5`
- US growth-heavy equity: `phi` around `0.2`
- concentrated US equity: `phi` can drift toward `1.0`
- broad or heterogeneous universes: low `phi`

## Current Practical Reading

If the goal is to keep developing the Eq. 8 workflow around `ARP_AGNOSTIC`,
the most credible current working hypothesis is:

- keep `phi` configurable by market family,
- do not hard-code `phi=0.5` globally,
- use intermediate `phi` as the default research prior for equity universes,
- stay cautious with high `phi` outside concentrated equity markets.

## Sensitivity To Covariance Window Scaling

A second pass was run with:

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

- for `dji` and `world_index`, the scaled window fell below the baseline
  `covariance_min_periods = 60`
- these two universes were therefore rerun with
  `covariance_min_periods = covariance_window`
- this is not a cosmetic detail; without it, the protocol is not computable and
  the runs collapse to zero

### What changed little

- `cac40` remained clearly hump-shaped, with the best region still around
  `phi=0.4-0.5`
- `eurostoxx50` also kept the same qualitative conclusion, again with a best
  region around `phi=0.4-0.5`
- `nasdaq100` still preferred an intermediate `phi`, with the optimum staying
  near `phi=0.2`

Representative rerun results:

- `cac40`, `window=60`: best `phi=0.4`, Sharpe `0.8634`
- `eurostoxx50`, `window=75`: best `phi=0.5`, Sharpe `1.0044`
- `nasdaq100`, `window=152`: best `phi=0.2`, Sharpe `1.4466`

Interpretation:

- these three markets are the strongest evidence that the original conclusions
  were not just artifacts of a single fixed `covariance_window=150`

### What changed materially

- `dji` no longer preferred `phi=1.0` once the window was reduced to `45`
  and made feasible with `min_periods=45`
- it now shows a shallow intermediate optimum around `phi=0.3`
- `world_index` no longer prefers `phi=0.0` under the scaled window protocol;
  it moves to a very mild optimum around `phi=0.5`
- `dataset_all` flips from low `phi` preference to a weak preference for
  `phi=1.0`

Representative rerun results:

- `dji`, `window=45`, `min_periods=45`: best `phi=0.3`, Sharpe `1.2617`
- `world_index`, `window=26`, `min_periods=26`: best `phi=0.5`, Sharpe `0.5864`
- `dataset_all`, `window=84`: best `phi=1.0`, Sharpe `0.6518`

Interpretation:

- the broad or heterogeneous universes are much more window-sensitive than the
  main equity universes
- the core European and `nasdaq100` conclusions look robust
- the `dji`, `world_index`, and `dataset_all` stories should be treated as more
  conditional on the estimator horizon

### Best-Sharpe Delta: `window=150` vs `window=1.5N`

Best Sharpe by universe:

- `cac40`: `0.8565 -> 0.8634` (`+0.0069`)
- `eurostoxx50`: `1.0777 -> 1.0044` (`-0.0733`)
- `dji`: `1.2754 -> 1.2617` (`-0.0136`)
- `nasdaq100`: `1.4531 -> 1.4466` (`-0.0065`)
- `world_index`: `0.7046 -> 0.5864` (`-0.1183`)
- `dataset_all`: `0.5592 -> 0.6518` (`+0.0927`)

Reading:

- `cac40`, `dji`, and `nasdaq100` move only marginally.
- `eurostoxx50` weakens somewhat, but without changing the qualitative shape of
  the `phi` surface.
- `world_index` is strongly degraded by the shorter scaled window.
- `dataset_all` improves materially, which confirms that its results are highly
  protocol-dependent.

## Relationship With `ATF_AGNOSTIC`

Compared with the previous `ATF_AGNOSTIC` study:

- the `ARP_AGNOSTIC` results are easier to interpret because there is no trend
  horizon confounder,
- the evidence for intermediate `phi` is stronger here than in the TF case,
- this suggests that part of the instability seen in `ATF_AGNOSTIC` comes from
  the signal itself, not only from the `Q_phi` family.

## Suggested Next Steps

- Add direct plots of `Sharpe(phi)` for the equity universes.
- Test whether the best `phi` remains stable on rolling subperiods.
- Compare these `ARP_AGNOSTIC` surfaces with the corresponding `ATF_AGNOSTIC`
  surfaces market by market, to separate signal uncertainty from structural
  portfolio effects.
