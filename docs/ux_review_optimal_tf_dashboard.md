# UX Review - `optimal_tf` Dashboard

Initial review date: 2026-06-29
Updated review date: 2026-07-01
URL reviewed: `http://localhost:8502/`
Reference UX: `Standard / Strategy testbed`

## Update Summary - 2026-07-01

This document now includes:

1. The original coherence review from 2026-06-29
2. A follow-up review performed on 2026-07-01 after UX evolutions in the app

### Overall progression since the first review

The dashboard is noticeably more coherent than it was on 2026-06-29.

What improved:

- most operational services now share the same page grammar
- several screens now start with a short purpose statement
- contextual help is much more consistent
- recommendation blocks are now used more systematically
- `This run will:` summaries improve predictability before execution
- the `Tuning` family is much more aligned internally
- `Compare` and `Inspection snapshot` now feel like first-class members of the same product family

What still needs attention:

- service switching latency is still inconsistent across screens
- the most complex services still require more effort than the simpler ones
- `Config editor` is still structurally different from the rest
- date-widget helper text still adds visual noise

### Measured service-switch responsiveness on 2026-07-01

Observed transition times between service selection and visible heading update:

- around `65-200 ms`: `Guide`, `Evaluation`, `Vary frequency`
- around `350-515 ms`: `Vary cleaning`, `Vary window`
- around `900-1010 ms`: `Config editor`, `Allocation`, `Strategy testbed`, `Compare`, `Vary strategy`, `Hyperparameter tuning`, `Inspection snapshot`

Interpretation:

- the product is now conceptually more coherent
- but the feeling of fluidity is still uneven depending on the service
- this is now one of the main UX quality gaps because users can feel the inconsistency while navigating

---

## Follow-up Findings - 2026-07-01

### 1. Cross-service coherence is now materially better

Severity: Positive progress

Compared with the 2026-06-29 review, the dashboard now behaves more like a unified product:

- most services have a clear title
- most services now open with a one-line explanation
- most services expose recommendation patterns in a similar voice
- most services now include an execution summary with `This run will:`

This is the strongest improvement in the product since the first review.

### 2. Service-switch latency is now the main coherence problem

Severity: High

The dashboard is more coherent in structure and language, but it still does not feel equally responsive from one service to another.

Why this matters:

- users feel the product as one tool only if navigation feels consistent
- when some services update nearly instantly and others take around one second, the app feels uneven even if the layouts are aligned
- it affects both perceived quality and confidence

UX correction target:

- make service changes complete within a more consistent latency band
- if a service needs heavier setup, communicate it with a brief loading state rather than silent lag

### 3. `Strategy testbed` is still the best UX reference, but the gap is smaller now

Severity: Medium

`Strategy testbed` remains the most mature screen because it still does the best job of explaining:

- presets
- locked vs editable controls
- conditional parameter relevance

However, the gap has narrowed because other screens now explain:

- what the service is for
- when to use it
- what the run will do

This is a strong improvement and should be preserved.

### 4. The `Tuning` family is much more coherent than before

Severity: Positive progress

The `Tuning` services now behave like a real family:

- `Vary cleaning`
- `Vary window`
- `Vary strategy`
- `Vary frequency`
- `Hyperparameter tuning`

Shared strengths:

- similar experiment framing
- consistent use of recommendations
- better explanation of what is varied and what stays fixed

Remaining gap:

- `Hyperparameter tuning` is still visually and cognitively heavier than the others, even though its structure has improved

### 5. `Config editor` is improved but still sits outside the common UX rhythm

Severity: Medium

The screen is better introduced than before and now clearly explains that it is an administrative view over shared TOML defaults.

But it still differs sharply from the operational services because:

- it has far more sections
- it has many more fields visible at once
- it lacks the same operational rhythm of `purpose -> strategy -> parameters -> outcome -> run`

This is not necessarily wrong, but it still weakens the feeling of a single unified dashboard.

### 6. Ease of use is now good on simple services and mixed on dense services

Severity: Medium

Simple and mid-complexity services are now easier to use:

- `Allocation`
- `Evaluation`
- `Compare`
- `Inspection snapshot`
- `Vary cleaning`
- `Vary window`
- `Vary strategy`
- `Vary frequency`

The hardest service remains:

- `Hyperparameter tuning`

Reason:

- the user must absorb a large search space early in the screen
- the structure is better than before, but the first-read effort is still high

### 7. Global context in the sidebar is coherent but still heavy

Severity: Low

The sidebar now plays a more coherent product role, but it remains dense because many screens repeat:

- `Universe group`
- `Universe`
- `Start date`
- `Evaluation start`
- `Evaluation end`
- `Refresh prices now`

This is not a correctness problem, but it still slows down visual scanning for frequent users.

### 8. Date helper text remains noisier than the surrounding product language

Severity: Low

The product’s own explanatory text is now stronger and more deliberate. That makes the verbose calendar helper messages stand out even more:

- `Press the down arrow key to interact with the calendar...`
- `Selected date is ...`

These messages feel incidental compared with the much better domain guidance now present elsewhere.

---

## Goal

This document combines:

1. The UX review already performed on the dashboard
2. A prescriptive coherence review to help align all services with the strongest current UX pattern, which is `strategy testbed`

The objective is not to redesign the product from scratch. It is to make the different services feel like parts of the same product, with the same interaction grammar, the same level of guidance, and the same trust signals.

---

## Part 1 - Review Findings

### 1. Service switching is not atomically reflected in the UI

Severity: High

When the selected service changes in the sidebar, the main content can lag behind for around 2 to 3 seconds. During that interval, the UI can show a mismatched state:

- sidebar says `Selected Strategy testbed`
- main panel still shows `Standard / Compare`
- action button still shows `Run compare`

Why this matters:

- It breaks trust in the dashboard
- It creates uncertainty about which service will actually run
- It is especially risky in a control surface where users launch computations or evaluations

UX correction target:

- service selection and main panel content should update as a single perceived transition
- if loading is unavoidable, show a temporary loading state for the main panel and disable the run button until the new service is ready

### 2. `Strategy testbed` is more mature than the rest of the product

Severity: Medium

`Strategy testbed` is currently the most pedagogical service:

- it explains the active preset
- it explains what is locked and what remains editable
- it provides conditional guidance for advanced parameters like `phi`, `trend span`, and `trend alpha`

Other services expose similarly technical controls but with far less guidance:

- `Allocation`
- `Evaluation`
- `Compare`
- `Vary cleaning`
- `Vary window`
- `Vary frequency`
- `Inspection snapshot`

Impact:

- the user experience feels uneven from one service to another
- advanced users can adapt, but the product does not feel deliberately designed as a coherent suite

### 3. Terminology and labels are not fully standardized

Severity: Medium

Examples observed:

- `Config defaults:` vs `Default from config:`
- `Strategy family` vs `Strategy families`
- `Strategy` vs `Strategies`
- `Covariance window` vs `Window` vs `Windows`
- `Methods` vs `Cleaning methods`

Impact:

- users have to re-parse the UI instead of reusing the same mental model
- the dashboard feels assembled service by service rather than designed as one system

### 4. `Config editor` follows a different UX grammar from the rest

Severity: Medium

Most services follow a fairly stable pattern:

- context and scope in the sidebar
- title in the main panel
- `Strategy`
- `Service parameters`
- one final action

`Config editor` instead becomes a large configuration form with many sections:

- `Universe`
- `Estimation`
- `Backtest`
- `Allocation / Evaluation`
- `Compare`
- `Output`

Impact:

- this makes sense functionally, but it feels disconnected from the rest of the dashboard
- the user moves from "service operation" UX to "raw configuration administration" UX without a strong transition

### 5. Complex services do not always receive proportionate guidance

Severity: Low

Services like `Compare` and `Hyperparameter tuning` expose broader combinatorics than simpler services, but they are not explained with the same care as `Strategy testbed`.

Impact:

- complexity grows, but guidance does not scale with it
- users need to infer more than they should

### 6. Date widget helper text is visually noisy

Severity: Low

Several services surface verbose date-picker accessibility text directly in the reading flow, for example:

- `Press the down arrow key to interact with the calendar...`
- `Selected date is ...`

Impact:

- these messages interrupt scanning of the form
- they visually compete with more meaningful domain guidance

---

## Part 2 - Prescriptive Coherence Review

### Product principle

The dashboard should feel like one operating console with multiple service modes, not like a collection of separate forms.

The best current model is `strategy testbed` because it already does three important things well:

1. It explains what the current selection means
2. It explains which controls matter and when
3. It keeps structure stable while allowing advanced configuration

Every other service should inherit that same UX grammar.

---

## Target UX Contract For All Services

Each service should follow the same high-level structure.

### 1. Stable main-panel anatomy

Recommended order:

1. Service title
2. One-sentence purpose
3. `Strategy` section when relevant
4. `Service parameters` section
5. Optional `Advanced options` section
6. Context/help block with defaults and recommendations
7. Final primary action

What this fixes:

- faster orientation
- lower mental switching cost between services
- more consistent visual rhythm

### 2. Stable sidebar anatomy

Recommended order:

1. `Usage mode`
2. `Service`
3. short service description
4. `Config path`
5. shared market/backtest context controls only if truly global

Guideline:

- keep the sidebar focused on navigation and cross-service context
- keep service-specific logic in the main panel

### 3. Stable language conventions

Recommended standards:

- always use `Strategy family`
- always use `Strategy` for a single choice
- always use `Strategies` for multi-select
- always use `Cleaning method` for a single choice
- always use `Cleaning methods` for multi-select
- always use `Covariance window` for one value
- always use `Covariance windows` for a list
- always use `Config defaults:` for summary hints

This should become a dashboard-wide naming policy.

### 4. Stable help patterns

Use exactly three layers of help:

- inline field labels
- short contextual hints below a section or control group
- one compact recommendation block near the action

Avoid:

- mixing defaults, warnings, and domain explanations everywhere
- making every service invent its own hint style

---

## Service-By-Service Alignment Recommendations

### Guide / Strategy guide

Current state:

- clear and readable
- acts as documentation rather than operation

Recommendation:

- keep it lightweight
- use it as the reference for conceptual explanations
- add links or references from operational services back to this guide when users need deeper theory

Coherence goal:

- conceptual help lives here
- operational guidance in services should stay shorter and more task-oriented

### Config / Config editor

Current state:

- functionally rich
- structurally much denser than the rest
- does not feel like the same product mode

Recommendations:

- add a short intro block explaining what this editor changes globally
- group sections more clearly into `Market data`, `Estimation`, `Backtest`, `Service defaults`, `Outputs`
- consider a compact summary panel at the top with the current config identity
- make save/apply behavior explicit if not already obvious

Coherence goal:

- this can remain more administrative than the other services
- but it should still use the same hierarchy and guidance style

### Standard / Allocation

Current state:

- structurally clean
- lower guidance density than `strategy testbed`

Recommendations:

- add a one-line explanation of what this service returns
- add a short hint explaining when to use allocation instead of evaluation
- align the defaults/help block format with `strategy testbed`

Coherence goal:

- same clean shell as today
- better framing of intent and expected output

### Standard / Evaluation

Current state:

- close to allocation
- understandable, but still less guided than `strategy testbed`

Recommendations:

- add a compact explanation of what changes compared to allocation
- explain `rebalance frequency` and `weight smoothing alpha` in the same tone as `strategy testbed`
- surface a short “typical use case” note

Coherence goal:

- position it clearly as the packaged backtest path

### Standard / Strategy testbed

Current state:

- strongest UX in the product today
- best use of explanatory hints
- best balance between flexibility and orientation

What to preserve:

- preset explanation
- locked-vs-editable explanation
- conditional hints for advanced parameters
- clean sectioning

What to reuse elsewhere:

- explanatory microcopy pattern
- advanced parameter disclosure logic
- confidence-building defaults summary

### Standard / Compare

Current state:

- coherent structure
- higher complexity than allocation/evaluation
- less explanation than its complexity deserves

Recommendations:

- add a short explanation of what is being compared and what output to expect
- explain strategy multi-selection more clearly
- mirror the `strategy testbed` style for complex parameter interactions

Coherence goal:

- should feel like a comparison-oriented sibling of `evaluation`, not a separate tool

### Tuning / Vary cleaning

Current state:

- compact and understandable
- acceptable structure

Recommendations:

- explain the objective of the experiment in one line
- rename and standardize help text around methods
- use the same recommendation block layout as standard services

Coherence goal:

- make it feel like an experiment template, not just a stripped-down form

### Tuning / Vary window

Current state:

- structurally consistent
- under-explained

Recommendations:

- explain what the user learns by varying windows
- standardize the naming to `Covariance windows`
- add one hint on choosing a sensible window range

Coherence goal:

- same skeleton as `vary cleaning`, but with task-specific framing

### Tuning / Vary strategy

Current state:

- useful service
- conceptually heavier because it compares strategy families

Recommendations:

- add explicit framing of what is being held constant and what is being varied
- standardize `Strategy family` vs `Strategy families` usage
- make result expectations clearer before the user runs it

Coherence goal:

- closer to `compare`, but with a tuning mindset

### Tuning / Vary frequency

Current state:

- multi-value experimental service
- more complex than the base tuning forms

Recommendations:

- add a short note explaining the operational effect of testing multiple rebalance frequencies
- group frequency choices visually as a comparison set
- align defaults and recommendations with the other tuning screens

Coherence goal:

- consistent experiment framing

### Tuning / Hyperparameter tuning

Current state:

- one of the most complex screens
- many candidate strategies, methods, windows, and frequencies
- not enough framing relative to its complexity

Recommendations:

- add a top summary block: what is optimized, over which search space, and what output the user gets
- split controls into `Search space`, `Backtest context`, and `Output`
- consider progressive disclosure for advanced options to reduce first-load density

Coherence goal:

- this should feel like the advanced sibling of the tuning family, not like a raw parameter wall

### Inspection / Inspection snapshot

Current state:

- close to evaluation in structure
- overall coherent

Recommendations:

- add a one-line explanation of what an inspection snapshot contains
- explain why the inspection date is different from evaluation dates
- align help text format with the standard services

Coherence goal:

- should feel like a diagnostic view of the same workflow, not a separate subsystem

---

## Coherence Gaps Relative To `Strategy testbed`

These are the main qualities from `strategy testbed` that should be generalized.

### A. Explain the current configuration state

`Strategy testbed` says:

- what preset is active
- what that preset implies
- what is locked

Recommendation:

- every service should expose a short state summary when the selected combination materially affects the form

### B. Explain conditional parameter relevance

`Strategy testbed` already explains:

- when `trend span` matters
- when `phi` matters

Recommendation:

- reuse this pattern anywhere a control is context-dependent

### C. Use consistent confidence signals

`Strategy testbed` combines:

- config defaults
- domain recommendation
- task-oriented framing

Recommendation:

- every operational service should use the same trio:
  - what defaults are active
  - what recommendation applies
  - what the action will run

### D. Match guidance density to task complexity

Simple services:

- should stay short and clean

Complex services:

- should earn more explanatory scaffolding

Recommendation:

- do not give every service the same amount of text
- give every service the same quality of explanation

---

## Priority Fixes

### P0

- Fix the delayed service-content synchronization
- Disable or guard action buttons while a new service view is still loading

### P1

- Standardize terminology across all services
- Reuse `strategy testbed` guidance patterns in `Evaluation`, `Compare`, `Hyperparameter tuning`, and `Inspection snapshot`
- Normalize defaults/help blocks across the dashboard

### P2

- Refactor `Config editor` to better fit the product-wide hierarchy
- Reduce the visual noise of date-picker helper text
- Add clearer top-level explanations for tuning services

---

## Suggested UX Checklist For Corrections

Use this checklist while implementing fixes.

### For every service

- Does the title match the selected service immediately?
- Is the primary action label aligned with the title?
- Is there a one-sentence explanation of the service purpose?
- Are section names consistent with the rest of the dashboard?
- Are defaults shown in the same format as elsewhere?
- Are advanced or conditional parameters explained?
- Does the screen show what is being varied, compared, or optimized?

### For cross-service coherence

- Do all services reuse the same terminology?
- Do complex screens provide more guidance than simple screens?
- Is the same visual hierarchy used everywhere?
- Does the dashboard feel like one tool with multiple modes?

---

## Recommended Implementation Strategy

1. Fix service switching trust issues first
2. Create a shared microcopy and labeling standard
3. Extract a common help-pattern for all operational services
4. Align the most visible services first:
   - `Evaluation`
   - `Compare`
   - `Inspection snapshot`
   - `Hyperparameter tuning`
5. Rework `Config editor` once the shared grammar is stable

---

## Final Take

The dashboard already has a strong structural base. The main issue is not that each screen is bad on its own. The issue is that `strategy testbed` shows what the product can feel like when complexity is explained well, and the other services have not fully caught up yet.

The best path forward is to treat `strategy testbed` as the reference interaction model for:

- guidance density
- explanatory microcopy
- parameter framing
- confidence-building around execution

If the rest of the dashboard is aligned to that standard, the product will feel significantly more coherent without needing a full redesign.
