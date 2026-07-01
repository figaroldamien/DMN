# `optimal_tf` Product Priorities

Date: 2026-07-01

## Goal

Define the next product decisions in the right order, so the dashboard evolves from a useful multi-service UI into a coherent research and execution product.

---

## Priority 1 - Make navigation feel reliable

Decision:

- reduce the variance of service-switch latency
- show a clear loading state when a screen needs time to update

Why now:

- this affects every user, every session
- inconsistent transitions reduce trust in the product

Expected gain:

- stronger perceived quality
- better confidence before launching runs

Do not do too late:

- adding more services before this is stabilized

---

## Priority 2 - Reorganize the product by user intent

Decision:

- move away from a taxonomy centered on `Standard`, `Tuning`, `Inspection`, `Config`
- reorganize around intents like `Workspace`, `Run`, `Compare`, `Search`, `Guide`

Why now:

- the current product is large enough that structure now matters more than adding one more screen

Expected gain:

- easier onboarding
- easier memorization
- lower navigation cost

Do not do too early:

- large visual redesign before agreeing on the target information architecture

---

## Priority 3 - Merge comparison workflows into one family

Decision:

- group `Compare`, `Vary strategy`, `Vary cleaning`, `Vary window`, and `Vary frequency` into one comparison family or one `Comparison Lab`

Why now:

- these services are conceptually close
- they currently consume too many separate mental slots

Expected gain:

- clearer product story
- simpler service discovery
- more reuse in the UI

Do not do too late:

- letting each comparison service diverge further in UX and behavior

---

## Priority 4 - Preserve `Strategy testbed` as the reference UX

Decision:

- keep `Strategy testbed` as the benchmark for guidance quality
- reuse its best patterns elsewhere

Why now:

- it already demonstrates the strongest balance between flexibility and usability

Expected gain:

- consistent product voice
- better support for complex settings

Do not do wrong:

- flattening this service just to make everything uniform

---

## Priority 5 - Separate workspace administration from execution

Decision:

- clearly isolate shared setup and config tasks from run-oriented tasks

Why now:

- `Config editor` still feels different from operational services
- users should always know whether they are editing the workspace or executing an analysis

Expected gain:

- cleaner mental model
- lower risk of confusion between persistent defaults and one-off run parameters

Do not do too late:

- piling more administrative controls into operational screens

---

## Priority 6 - Improve dense services before adding new ones

Decision:

- continue simplifying the hardest screens, especially `Hyperparameter tuning`

Why now:

- this is where usability still drops the most

Expected gain:

- broader usability beyond expert users
- better scalability of the product surface

Do not do too early:

- building advanced recommendation layers before the core dense screens are readable

---

## Priority 7 - Add result-reading capabilities

Decision:

- invest in reading and comparing past runs, not only launching new ones

Best candidate:

- a `Results explorer`

Why now:

- the product is becoming a research tool
- research tools need memory, not just execution

Expected gain:

- faster iteration
- better reuse of prior work
- stronger decision support

Do not do too late:

- otherwise the product remains a launcher rather than a learning system

---

## Priority 8 - Add scenario-oriented workflows

Decision:

- let users compare named scenarios, not only parameter sets

Best candidate:

- `Scenario compare`

Why now:

- users often think in strategies or scenarios, not in isolated control values

Expected gain:

- much better readability
- easier communication of findings

Do not do too early:

- before the comparison family is structurally clean

---

## Priority 9 - Add standard run summaries

Decision:

- generate a short summary artifact after each run

Possible contents:

- key inputs
- what changed
- main outputs
- warnings
- suggested next step

Why now:

- execution is already strong enough to justify better interpretation support

Expected gain:

- better user confidence
- easier review of results
- easier collaboration

---

## Priority 10 - Add a workflow recommender last

Decision:

- only later, add a layer that helps users choose the right service from their intent

Why later:

- this becomes valuable only when the product structure beneath it is already stable

Expected gain:

- lower onboarding cost
- easier service discovery for less frequent users

Do not do too early:

- otherwise it becomes a workaround for a structure that is not yet clean enough

---

## Recommended Order

1. Stabilize service-switch reliability
2. Validate the new product information architecture
3. Merge comparison workflows conceptually
4. Isolate workspace administration
5. Improve dense screens
6. Build result-reading capabilities
7. Add scenario workflows
8. Add run summaries
9. Add recommendation / assistant layers

---

## What Not To Do Next

- Do not add many new services before regrouping the existing ones
- Do not redesign the visuals before deciding the product structure
- Do not hide complexity with assistant features before simplifying the core workflows
- Do not treat `Config editor` as just another operational screen

---

## Final Product Thesis

The best next step is not “more tools”.

The best next step is:

- fewer top-level concepts
- clearer user-intent grouping
- stronger continuity between services
- better support for comparing, understanding, and reusing results

That is what will turn `optimal_tf` from a strong dashboard into a strong product.
