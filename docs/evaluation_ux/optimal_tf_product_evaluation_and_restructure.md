# Product Evaluation - `optimal_tf` Dashboard

Date: 2026-07-01
Scope: product evaluation, restructuring options, and new service proposals

## Executive Summary

`optimal_tf` is no longer just a technical dashboard for launching isolated computations. It is becoming a research and decision product for portfolio construction workflows.

The product is already strong on:

- exposing domain capabilities in a usable way
- distinguishing run, comparison, tuning, and inspection tasks
- making advanced workflows accessible from one surface

The next product step is not mainly to add more parameters. It is to improve the product architecture so users navigate by intent, not by implementation category.

Current product risk:

- the dashboard still partly reflects backend or service taxonomy
- user intent is visible, but not yet the main organizing principle

Recommended direction:

- evolve from a `multi-service dashboard` into a `research and execution workspace`

---

## Product Assessment

### What the product already does well

#### 1. It covers the full decision loop

The current product already supports most of the core workflow:

- set defaults and context
- run a strategy
- compare alternatives
- inspect one state in depth
- search over a broader parameter space

This is a strong foundation because many internal tools stop at either:

- raw execution
- or raw diagnostics

`optimal_tf` already connects both.

#### 2. It has emerging product families

The dashboard is beginning to show coherent families of use:

- direct execution
- controlled comparison
- structured experimentation
- diagnosis

That means the product now has enough shape to support a more intentional information architecture.

#### 3. It is becoming pedagogical

The app is no longer only a control panel. It increasingly explains:

- why to use a service
- what it varies
- what the run will do

That shift is important because it lowers the activation cost for non-authors and occasional users.

---

## Main Product Weaknesses

### 1. The product is still organized more by implementation mode than user job

Today, the user still has to interpret categories like:

- `Standard`
- `Tuning`
- `Inspection`
- `Config`

These make sense internally, but they are not the clearest expression of user intent.

Example:

- `Compare`
- `Vary strategy`
- `Vary cleaning`
- `Vary window`
- `Vary frequency`

All belong to one user job: compare alternatives under shared assumptions.

But they are split between different conceptual buckets.

### 2. Some services are too close conceptually to justify separate mental slots

For a user, the difference between:

- `Compare`
- `Vary strategy`
- `Vary cleaning`
- `Vary window`
- `Vary frequency`

is often secondary to the higher-level task:

- “I want to understand what changes when I vary one factor”

The product currently asks users to think in service names before thinking in goals.

### 3. The product is stronger at launching than at capitalizing learning

Right now the product seems optimized for:

- setting up a run
- executing it

More than for:

- reviewing what was learned
- comparing historical runs
- surfacing reusable scenario results

This creates a ceiling for product maturity.

---

## Recommended Product Reorganization

## Principle

Organize the product around user intent:

- set up context
- run something
- compare alternatives
- search for better configurations
- inspect and understand

This makes the product easier to learn and easier to scale.

---

## Proposed Top-Level Product Architecture

### 1. `Workspace`

Purpose:

- define the shared environment for all downstream actions

Contents:

- `Config editor`
- default market / universe context
- shared dates and data freshness controls
- global execution defaults

Why:

- this isolates administrative and shared setup work
- it removes ambiguity between “editing the workspace” and “running a service”

### 2. `Run`

Purpose:

- execute one clearly defined analysis or production action

Contents:

- `Allocation`
- `Evaluation`
- `Inspection snapshot`

Why:

- these are single-target services
- they answer “run this one thing for me”

Suggested internal structure:

- `Allocation`: one-date output
- `Evaluation`: path / backtest output
- `Inspection snapshot`: one-date diagnostic output

### 3. `Compare`

Purpose:

- compare a small number of controlled alternatives

Contents:

- `Compare`
- `Vary strategy`
- `Vary cleaning`
- `Vary window`
- `Vary frequency`

Why:

- these are all forms of controlled comparison
- the user mental model is the same: hold most things fixed, vary one dimension

Suggested internal entry points:

- compare strategies
- compare cleaning methods
- compare covariance windows
- compare rebalance frequencies

### 4. `Search`

Purpose:

- explore a broader configuration space to find promising candidates

Contents:

- `Strategy testbed`
- `Hyperparameter tuning`

Why:

- these are exploratory services rather than standard operational runs
- they support research, not just execution

Suggested distinction:

- `Strategy testbed`: interactive focused sandbox
- `Hyperparameter tuning`: broader search/grid engine

### 5. `Guide`

Purpose:

- help users understand concepts, services, and outputs

Contents:

- `Strategy guide`
- parameter glossary
- interpretation notes for outputs
- decision help for choosing the right service

Why:

- this avoids overloading operational screens with too much theory

---

## Alternative Restructuring Option

If you want a more incremental change, keep the current top-level modes but refactor the content under them.

### Keep:

- `Config`
- `Standard`
- `Tuning`
- `Inspection`
- `Guide`

### Change:

- move `Compare` under a sub-family named `Comparison`
- group all `Vary ...` screens visually as `Comparison experiments`
- position `Strategy testbed` and `Hyperparameter tuning` as `Exploration`

This is less disruptive but also less strong from a product perspective.

---

## Services To Merge Or Reframe

### 1. Create a single `Comparison Lab`

Instead of exposing many separate comparison services at the same product level, expose one service shell with a first choice:

- compare strategies
- compare cleaning methods
- compare covariance windows
- compare rebalance frequencies

This would absorb:

- `Compare`
- `Vary strategy`
- `Vary cleaning`
- `Vary window`
- `Vary frequency`

Benefits:

- much clearer mental model
- shared UI shell
- lower navigation cost
- easier future expansion

### 2. Keep `Strategy testbed` separate

Reason:

- it is not just comparison
- it is a focused exploration sandbox
- it has a distinct value as the “interactive lab bench”

### 3. Keep `Hyperparameter tuning` separate but related

Reason:

- it is the broad-search counterpart to the testbed
- it belongs with testbed in an exploration/search family

---

## New Services Worth Considering

### 1. `Scenario compare`

Purpose:

- compare named scenarios instead of only raw parameter sets

Examples:

- `ARP default`
- `ARP weekly`
- `ARP cleaner alt`
- `Agnostic identity / gross`

Why it matters:

- users often think in scenarios, not in isolated control values
- scenario naming improves readability and communication

Value:

- easier decision-making
- easier result sharing

### 2. `Run summary`

Purpose:

- produce a concise standard summary after each run

Contents:

- key inputs used
- what was varied
- main metrics
- main warnings
- suggested next step

Why it matters:

- this turns output into a readable decision artifact

Value:

- strong usability gain
- helps occasional users
- improves reproducibility

### 3. `Results explorer`

Purpose:

- review past runs and compare them without relaunching everything

Potential features:

- filter by service
- filter by universe
- filter by strategy
- compare recent runs
- reopen run parameters

Why it matters:

- the product should help users learn from past work, not only create new work

Value:

- turns the app into a persistent research tool

### 4. `Service recommender`

Purpose:

- help users choose the right service from their goal

Example prompts:

- “I want the latest weights”
- “I want to compare several strategies”
- “I want to search broadly”
- “I want to inspect one rebalance date”

Why it matters:

- this becomes increasingly useful as the product surface grows

Value:

- lower onboarding cost

### 5. `Experiment builder`

Purpose:

- let the user define:
  - what stays fixed
  - what varies
  - what output matters

Then derive the right comparison or search workflow automatically.

Why it matters:

- it shifts the product from service selection to experiment design

Value:

- long-term strong differentiator

---

## Product Direction Recommendation

If choosing only one strategic direction, I would recommend:

- make the product center of gravity `Compare + Search + Results`

Why:

- `Allocation` and `Evaluation` are necessary, but relatively standard
- the strongest differentiating value is in helping users explore, understand, and compare portfolio design choices

In other words:

- not just “run optimal_tf”
- but “use optimal_tf as a research and decision system”

---

## Suggested Roadmap

### Short term

- improve top-level information architecture
- reduce service sprawl by regrouping comparison workflows
- keep refining cross-service coherence
- stabilize service-switch responsiveness

### Medium term

- introduce `Scenario compare`
- introduce `Run summary`
- introduce `Results explorer`

### Longer term

- evolve toward an experiment-centric workflow
- add recommender or builder layers that help users choose or compose the right workflow

---

## Recommended Product Positioning

The product should increasingly be framed as:

`A research and execution workspace for portfolio strategy design, comparison, and diagnosis`

This is stronger than:

- a config editor
- a service launcher
- a tuning panel

Because it reflects the actual value the product is starting to provide.

---

## Final Recommendation

The best immediate restructuring is:

1. Move from service taxonomy to intent taxonomy
2. Consolidate all “vary one factor” workflows into a shared comparison family
3. Treat `Strategy testbed` and `Hyperparameter tuning` as an exploration/search space
4. Separate workspace administration from operational execution
5. Start building result-reading capabilities, not just run-launching capabilities

If this direction is followed, the product will feel less like a dashboard of tools and more like a coherent research platform.
