# `optimal_tf` V2 - Product Requirements And Refactoring Plan

Date: 2026-07-02
Status: draft
Scope: product requirements for a V2 of the research experience around `optimal_tf_dashboard`, `portfolio_dashboard`, and comparison workflows

## 1. Product Intent

`optimal_tf` V2 should evolve from a service launcher into an integrated research workspace for:

- designing strategies
- running backtests
- understanding portfolio construction
- comparing scenarios
- identifying robust sources of edge

The main problem to solve is workflow fragmentation.

Today, a user can:

- run a strategy in `optimal_tf_dashboard`
- export a market fork snapshot
- open another app
- reload the snapshot in `portfolio_dashboard`

This works technically, but it creates too much friction for iterative research.

V2 should replace that with a more continuous workflow:

1. run a strategy or scenario
2. inspect performance
3. inspect the resulting portfolio directly
4. compare several portfolio outcomes
5. refine the hypothesis

---

## 2. Core Product Goals

### Goal 1 - Make `Run` flow directly into portfolio analysis

The user should be able to move from backtest outputs to portfolio analysis without:

- manually exporting a snapshot
- changing app as the primary path
- reloading the same context in a second surface

### Goal 2 - Make `Compare` useful for portfolio structure, not only for NAV

The comparison experience should support:

- performance comparison
- portfolio composition comparison
- turnover and implementation comparison
- contribution and concentration comparison

### Goal 3 - Help users identify real edge, not only local best scores

The product should support evaluation of:

- robustness
- sensitivity to parameter changes
- implementation realism
- portfolio stability

### Goal 4 - Preserve exportability and specialist views

`portfolio_dashboard` should remain useful as a dedicated expert surface, but it should no longer be required for the main research flow.

---

## 3. Functional Requirements

## 3.1 Integrated Run-To-Portfolio Workflow

### Requirement R1

Each relevant `Run` service should expose portfolio analysis inside `optimal_tf_dashboard`.

Priority targets:

- `Run / Allocation`
- `Run / Evaluation`
- `Run / Inspection snapshot`

### Requirement R2

Portfolio analysis should work directly from in-memory run results when possible.

The primary path should not require:

- writing a fork snapshot
- manually selecting a snapshot path
- changing application

### Requirement R3

Snapshot export should remain available as a secondary capability for:

- persistence
- sharing
- reopening in dedicated dashboards

### Requirement R4

`Run / Evaluation` should support at least two portfolio analysis levels:

- latest portfolio state
- rebalance history exploration

This implies the ability to inspect:

- current weights
- historical rebalance weights
- turnover path
- concentration evolution

---

## 3.2 Richer Compare Workflow

### Requirement C1

`Compare` should go beyond summary table + NAV and expose several result lenses.

Minimum target lenses:

- `Summary`
- `NAV`
- `Weights`
- `Overlap`
- `Turnover`
- `Contribution`

### Requirement C2

The user should be able to compare several portfolio outcomes for scenario families such as:

- different rebalance frequencies
- different covariance windows
- different cleaning methods
- different strategy families

### Requirement C3

Portfolio comparison should support a selected rebalance date or selected portfolio state when relevant.

This is especially important for:

- `Vary frequency`
- `Vary strategy`
- `Compare`

### Requirement C4

At least one comparison view should expose portfolio deltas between two scenarios.

Examples:

- weight difference by ticker
- top names added / removed
- concentration change
- turnover change

---

## 3.3 Strategy Edge Evaluation

### Requirement E1

The product should make robustness visible, not only peak performance.

Useful future outputs:

- local parameter sensitivity
- neighborhood stability
- robustness score
- ranking stability

### Requirement E2

The product should relate performance to implementation quality.

Relevant metrics include:

- turnover
- cost drag
- concentration
- overlap stability
- sector or bucket drift

### Requirement E3

The product should progressively support named scenarios rather than only raw parameter sets.

Examples:

- `ARP default`
- `ARP weekly`
- `ARP smoother`
- `ATF trend fast`
- `Agnostic phi 0.5 monthly`

This is important for readability, communication, and reuse.

---

## 3.4 Product Surface Boundaries

### Requirement B1

`optimal_tf_dashboard` should become the primary research workspace.

Its role:

- configure
- run
- compare
- inspect portfolio consequences

### Requirement B2

`portfolio_dashboard` should become a specialized portfolio exploration surface, not the mandatory next step after every run.

Its role:

- deep portfolio analysis
- persistent snapshot reopening
- expert use cases

### Requirement B3

The system should use a shared analysis core so that portfolio analysis logic is not duplicated across apps.

---

## 4. UX Requirements

### UX1

The path from run execution to portfolio understanding should feel continuous.

### UX2

Portfolio analysis should appear as a natural result tab or result section, not as an external workflow branch.

### UX3

`Compare` should explain not only which scenario wins, but why it wins structurally.

### UX4

When multiple portfolios are compared, the UI should make it easy to switch between:

- performance lens
- implementation lens
- composition lens

### UX5

Advanced analysis should remain progressive:

- quick summary first
- deeper structural views second

---

## 5. Engineering Principles

### Principle 1 - Build a shared portfolio analysis core

Do not improve the integration by wiring apps together more tightly at the UI level only.

Instead, extract reusable portfolio analysis components that can be fed from:

- an in-memory run result
- a persisted snapshot

### Principle 2 - Separate context building from rendering

We should distinguish:

- context builders
- analysis computations
- rendering components

This will make reuse across apps and services easier.

### Principle 3 - Preserve backward compatibility for existing snapshots

Existing fork snapshot workflows should continue to work during the transition.

### Principle 4 - Optimize for incremental delivery

This V2 should be built in small refactoring lots, not through a big-bang rewrite.

---

## 6. Refactoring Plan

## Lot A - Extract shared portfolio analysis engine

Objective:

- move reusable logic out of `portfolio_dashboard` into shared `src/optimal_tf/...` components

Scope:

- context builders from run result
- context builders from snapshot
- shared portfolio summary helpers
- shared NAV / drawdown / holdings / sleeve analysis helpers

Deliverables:

- a reusable portfolio analysis module
- no mandatory UI changes yet

Why first:

- this is the enabling layer for the rest

Risks:

- accidental duplication if extraction is partial
- regressions in `portfolio_dashboard` if compatibility is not preserved

Success criteria:

- `portfolio_dashboard` still works
- the same analysis core can be called from `optimal_tf_dashboard`

## Lot B - Add `Portfolio analysis` to `Run`

Objective:

- integrate portfolio analysis directly into `Run / Allocation`, `Run / Evaluation`, and `Run / Inspection snapshot`

Scope:

- new result tabs or sub-tabs
- latest portfolio view
- selected rebalance date view where available
- keep snapshot export as optional secondary action

Deliverables:

- in-app run-to-portfolio workflow

Why second:

- highest user value with limited conceptual expansion

Risks:

- result pages becoming too dense
- performance overhead if heavy data is recomputed repeatedly

Success criteria:

- the user can inspect portfolio structure after a run without leaving the app

## Lot C - Enrich `Compare` with structural portfolio views

Objective:

- make `Compare` useful for understanding portfolio differences, not only performance differences

Scope:

- `Weights` tab
- `Overlap` tab
- `Turnover` tab
- `Contribution` tab
- selected date / rebalance state comparison

Deliverables:

- multi-portfolio comparison tools inside the `Compare` family

Why third:

- this creates the strongest product differentiation for strategy research

Risks:

- UI overload
- large comparison sets becoming unreadable

Success criteria:

- frequency, window, cleaning, and strategy comparisons can be interpreted structurally

## Lot D - Introduce named scenarios

Objective:

- improve readability and reuse of experiments

Scope:

- scenario labels
- scenario metadata
- scenario-aware comparison displays

Deliverables:

- clearer research objects than raw parameter tuples

Why fourth:

- easier once run, compare, and portfolio analysis are already connected

Risks:

- scenario naming conventions becoming inconsistent if added too early

Success criteria:

- users can compare and discuss scenario names, not only raw parameter combinations

## Lot E - Add robustness and edge-evaluation tools

Objective:

- move from “best score finder” toward “robust edge evaluator”

Scope:

- sensitivity views
- local stability views
- robustness indicators
- performance vs implementation trade-off views

Deliverables:

- advanced research support for parameter robustness

Why fifth:

- depends on the richer comparison and scenario foundations

Risks:

- noisy signals if robustness metrics are added before the base workflow is clean

Success criteria:

- users can judge whether a result is strong, fragile, or implementation-dependent

---

## 7. Recommended Delivery Order

1. Extract shared portfolio analysis core
2. Integrate portfolio analysis into `Run`
3. Enrich `Compare` with structural portfolio views
4. Introduce scenarios as first-class research objects
5. Add robustness / edge-evaluation tools

---

## 8. What Not To Do

- Do not start by tightly coupling two Streamlit apps through manual navigation shortcuts only.
- Do not build a large `Results explorer` before the run-to-portfolio workflow is fluid.
- Do not overload `Compare` with too many structural views before shared reusable components exist.
- Do not remove snapshot export compatibility early in the refactor.

---

## 9. V2 Product Positioning

Target positioning:

`optimal_tf` V2 should feel like an integrated research and portfolio-understanding workspace for strategy design, comparison, and implementation-aware evaluation.

It should no longer feel like:

- one app to run
- one app to inspect
- one export bridge between them

It should feel like one coherent research loop.
