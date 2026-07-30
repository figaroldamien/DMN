# Strategy Evolution Log

Last updated: 2026-06-11

## Purpose

This document is the running journal for material strategy evolutions in
`optimal_tf`.

It should be updated whenever we:
- add a strategy
- materially change an existing strategy
- change a core estimation or evaluation convention that alters strategy interpretation
- decide to pause, reject, or defer a strategy idea after analysis

See also:
- [optimal_tf_strategies.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_strategies.md) for the current-state description of strategies
- [optimal_tf_specifications.md](/Users/damien.figarol/trading_app_lab/docs/reference/optimal_tf_specifications.md) for the functional contract
- [adr/0003-strategy-evolution-documentation-workflow.md](/Users/damien.figarol/trading_app_lab/docs/adr/0003-strategy-evolution-documentation-workflow.md) for the documentation rule

## Entry Template

Copy this block for each new entry.

```md
## YYYY-MM-DD - Short title

- Area: `strategy_name` or `shared_estimation` or `evaluation`
- Status: proposed | accepted | rejected | deferred | implemented
- Motivation:
- Decision:
- Why this choice:
- Expected impact:
- Validation:
- Follow-up:
```

## Entries

## 2026-06-11 - Weight smoothing moved to shared backtest config and implemented

- Area: `evaluation`
- Status: implemented
- Motivation: the first documentation pass exposed weight smoothing under `[evaluation]`, but the parameter is conceptually a portfolio implementation control, closer to transaction costs and volatility targeting than to scenario selection.
- Decision: move `weight_smoothing_alpha` into shared backtest configuration, make the loader accept `[portfolio]` as an alias of `[backtest]`, and apply smoothing inside the shared periodic backtest engine on implemented weights.
- Why this choice: the parameter now belongs to the common portfolio implementation layer and is therefore available consistently across services that reuse the shared backtest engine, rather than looking tied to one evaluation surface.
- Expected impact: keeps the Eq. 8 target-generation logic unchanged while reducing turnover through a shared implementation overlay; also makes the config surface more semantically coherent.
- Validation: targeted config-loading and evaluation tests were updated; the new smoothing behavior is covered by a dedicated backtest test.
- Follow-up: decide later whether `[backtest]` should be formally renamed to `[portfolio]` everywhere, now that the loader can already accept both names.

## 2026-06-11 - Turnover-control research logged with Eq. 8 smoothing preference

- Area: `evaluation`
- Status: proposed
- Motivation: the current rebalancing engine applies newly computed weights at each rebalance date without using the previously implemented portfolio as an input, which can create unstable weights, excessive turnover, and avoidable transaction costs.
- Decision: record the bibliography and frame the preferred research direction as a lightweight implementation overlay on top of Eq. 8, with simple weight smoothing as the first candidate.
- Why this choice: most current strategies rely on the closed-form Eq. 8 family, so a constrained optimizer with turnover terms would be a much larger methodological departure than desired. A smoothing layer preserves the core strategy definition while introducing memory of prior holdings.
- Expected impact: reduces weight instability and transaction costs while keeping the existing strategy family, parameter studies, and cross-market comparisons interpretable.
- Validation: no implementation yet; this step documents the literature review and the current project-level research stance.
- Follow-up: if this path is implemented, start with exponential smoothing of implemented weights, then compare net performance and turnover metrics against the current baseline before exploring deadbands or turnover caps.

## 2026-06-11 - Weekly correlation / covariance estimation idea deferred

- Area: `shared_estimation`
- Status: deferred
- Motivation: investigate whether weekly data for covariance / correlation cleaning would improve matrix stability relative to the current daily estimator.
- Decision: do not implement this evolution for now.
- Why this choice: the expected value was not clear enough relative to the added complexity, and the approach looked especially fragile for large universes because weekly sampling sharply reduces effective sample size.
- Expected impact: avoids introducing a risk-model branch whose main benefit might only be cosmetic stability rather than robust portfolio improvement.
- Validation: code-path review of the current estimator and backtest pipeline; no implementation was carried out.
- Follow-up: revisit only if we identify a concrete symptom such as unstable weights, excessive turnover, or clearly noisy correlation structure that weekly sampling could target.
