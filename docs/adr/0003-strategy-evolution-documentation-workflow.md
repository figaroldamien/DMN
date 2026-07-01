# ADR 0003: Document Strategy Evolutions Explicitly

Date: 2026-06-11

## Status

Accepted

## Context

Strategy work in `optimal_tf` often mixes:
- research hypotheses
- implementation choices
- simplifications relative to papers or prior ideas
- validation results
- decisions to continue, pause, or abandon a path

Without a durable trace, it becomes hard to answer basic questions later:
- why was this strategy change made?
- what alternatives were considered?
- was the goal performance, robustness, interpretability, or UX?
- was the idea accepted, rejected, or only parked?

The repository already contains strategy notes and ADRs, but there is no
explicit workflow requiring that strategy evolutions be logged as they happen.

## Decision

From now on, each material strategy evolution must be documented in the repo.

Required documentation workflow:
- log the change in `docs/strategy_evolution_log.md`
- update `docs/optimal_tf_strategies.md` when the public behavior, meaning, or positioning of a strategy changes
- add or update an ADR when the change introduces a durable architectural or methodological decision that should remain stable across future iterations

A "material strategy evolution" includes:
- adding a new strategy
- changing the portfolio construction logic of an existing strategy
- changing signal definition, normalization, leverage interpretation, or risk model conventions
- changing evaluation conventions when they materially affect strategy interpretation
- explicitly deciding not to pursue a researched strategy path after analysis

Each log entry should capture at least:
- date
- strategy or area concerned
- motivation
- change or decision taken
- expected impact
- validation status
- follow-up or open questions

## Consequences

Benefits:
- keeps strategy research auditable
- preserves decision rationale, not just final code
- makes it easier to compare implemented behavior with the intended design
- reduces re-litigation of earlier experiments

Trade-offs:
- every strategy PR or coding session has a small documentation cost
- some entries will describe decisions not to proceed, which adds history but also useful context

## Next Steps

Immediate project convention:
1. use `docs/strategy_evolution_log.md` as the running journal
2. keep `docs/optimal_tf_strategies.md` as the current-state reference
3. reserve ADRs for decisions that shape the long-term project structure or methodology
