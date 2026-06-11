# Turnover And Weight Smoothing Note

Last updated: 2026-06-11

## Purpose

This note records a short literature review on turnover control and transaction
costs, and translates it into a practical research direction for the
Eq. 8-based strategy family used in `optimal_tf`.

The immediate question is not whether turnover matters in theory. It does. The
real question is which kind of turnover control is compatible with the current
strategy design.

## Current Project Constraint

Many current strategies are built from the closed-form Eq. 8 family:

`w_raw = omega * C^{-1/2} * Q^{-1/2} * s`

For the flat-signal agnostic family discussed elsewhere, this becomes:

`w_raw = omega * C^{-1/2} * Q_phi^{-1/2} * 1`

with:

`Q_phi = phi * C + (1 - phi) * I`

This matters because a large part of the current research workflow relies on:

- an explicit closed-form mapping from estimated matrices to target weights,
- fast cross-market comparisons,
- interpretable parameter sweeps such as `phi`,
- a shared evaluation engine that consumes target weights after the fact.

Because of this design, replacing Eq. 8 with a constrained optimizer that
jointly solves for return, risk, turnover, and costs would be a much larger
methodological change than a simple strategy refinement.

## Main Reading

### 1. Transaction costs change the optimal trading policy

Theoretical work consistently shows that once transaction costs are introduced,
the optimal policy is no longer to rebalance exactly to the frictionless target
at every decision date.

Typical consequences are:

- no-trade regions,
- partial adjustment toward the target,
- explicit trade-off between tracking error and trading costs.

Useful references:

- Kallsen, Li (2013), *Portfolio Optimization under Small Transaction Costs: a Convex Duality Approach*  
  https://arxiv.org/abs/1309.3479
- Liu, Muhle-Karbe, Weber (2014/2017), *Rebalancing with Linear and Quadratic Costs*  
  https://arxiv.org/abs/1402.5306
- Cai, Judd, Xu (2020), *Numerical Solution of Dynamic Portfolio Optimization with Transaction Costs*  
  https://arxiv.org/abs/2003.01809

Practical reading for this project:

- the raw Eq. 8 weight should be interpreted as a frictionless target,
- the implemented tradable weight can reasonably differ from that target.

### 2. Higher rebalance frequency makes the issue more important

The literature also supports the intuition that the value of turnover control
increases when rebalancing becomes more frequent.

Useful reference:

- Ekren, Liu, Muhle-Karbe (2015/2017), *Optimal Rebalancing Frequencies for Multidimensional Portfolios*  
  https://arxiv.org/abs/1510.05097

Practical reading for this project:

- if we keep weekly or monthly rebalancing experiments, turnover control is
  worth studying,
- if we ever move to higher-frequency re-estimation, it becomes even more
  important.

### 3. Turnover penalties often act like useful regularization

Empirical and methodological papers often find that adding transaction-cost
awareness or turnover penalization improves out-of-sample behavior, not only by
reducing costs but also by stabilizing noisy allocations.

Useful references:

- Hautsch, Voigt (2019), *Large-Scale Portfolio Allocation Under Transaction Costs and Model Uncertainty*  
  https://arxiv.org/abs/1709.06296
- Boyd, Johansson, Kahn, Schiele, Schmelzer (2024), *Markowitz Portfolio Construction at Seventy*  
  https://arxiv.org/abs/2401.05080
- Fan, Medeiros, Yang, Yang (2024/2025), *Cost-aware Portfolios in a Large Universe of Assets*  
  https://arxiv.org/abs/2412.11575

Practical reading for this project:

- unstable weights are not only a cost problem,
- they are often also a symptom of estimation noise,
- a turnover-aware overlay can therefore improve robustness as well as net
  performance.

### 4. Cost model choice matters

The literature distinguishes between several types of costs:

- linear or proportional costs, for fees and spread-like effects,
- quadratic costs, for market impact and liquidity effects,
- mixed models, for more realistic execution modeling.

Useful reference:

- Chen, Lezmi, Roncalli, Xu (2020), *A Note on Portfolio Optimization with Quadratic Transaction Costs*  
  https://arxiv.org/abs/2001.01612

Practical reading for this project:

- the current engine already approximates proportional costs via turnover,
- this is good enough for a first turnover-control layer,
- there is no need to jump immediately to a richer optimizer.

## Project Conclusion

For `optimal_tf`, the most natural next step is not a constrained portfolio
optimizer inside the Eq. 8 core.

The most natural next step is to keep Eq. 8 as the raw target generator and add
a lightweight turnover-control overlay that maps:

`w_raw_t -> w_impl_t`

where:

- `w_raw_t` is the frictionless Eq. 8 target at rebalance date `t`,
- `w_impl_t` is the implemented tradable weight used by the backtest.

This keeps:

- the mathematical identity of the strategy,
- the current research workflow,
- comparability with earlier Eq. 8 experiments.

At the same time, it addresses:

- excessive turnover,
- unstable weights,
- overly reactive portfolio changes.

## Candidate Overlays Compatible With Eq. 8

### A. Exponential smoothing of weights

Simplest idea:

`w_impl_t = (1 - alpha) * w_impl_{t-1} + alpha * w_raw_t`

with `alpha in (0, 1]`.

Interpretation:

- `alpha = 1` gives the current behavior,
- lower `alpha` slows portfolio adjustment,
- the previous implemented portfolio becomes part of the decision rule.

Why this fits well:

- easy to implement,
- easy to explain,
- directly controls weight speed,
- does not alter the Eq. 8 formula itself.

Main caution:

- too small an `alpha` can make the portfolio lag genuine regime shifts.

### B. Deadband / no-trade threshold

Trade only when the gap between `w_raw_t` and `w_impl_{t-1}` is large enough.

Simple example:

- if `||w_raw_t - w_impl_{t-1}||_1 < tau`, do nothing,
- otherwise rebalance partially or fully.

Why this fits well:

- it is close in spirit to the no-trade region literature,
- it avoids paying costs for tiny weight changes caused by noise.

Main caution:

- threshold effects can make behavior less smooth and a bit less intuitive than
  pure exponential smoothing.

### C. Turnover cap by partial move to target

Move toward `w_raw_t`, but cap per-rebalance turnover.

Interpretation:

- retain the direction of the Eq. 8 signal,
- limit the amount of movement allowed at each rebalance.

Why this fits well:

- operationally robust,
- directly tied to the current turnover metric used by the engine.

Main caution:

- slightly more procedural than simple smoothing.

## Recommended Research Order

The lowest-risk research sequence is:

1. implement simple exponential smoothing of portfolio weights,
2. compare several `alpha` values against the current baseline,
3. inspect gross and net performance, average turnover, annualized turnover,
   and stability of holdings,
4. only then test a deadband or turnover cap if smoothing alone is not enough.

## Initial Working Hypothesis

The current best working hypothesis for this repository is:

- Eq. 8 should remain the portfolio construction core,
- turnover control should be added as a post-target implementation layer,
- simple weight smoothing is the best first candidate,
- constrained optimization can remain out of scope unless a future strategy
  branch intentionally departs from the Eq. 8 closed-form family.

## Validation To Run Later

When implemented, the following should be checked:

- change in average and annualized turnover,
- change in total transaction cost,
- change in net Sharpe and net return,
- sensitivity by universe,
- sensitivity by rebalance frequency,
- whether smoothing mostly reduces noise or also delays useful reallocations.

## References

- Boyd, Stephen; Johansson, Kasper; Kahn, Ronald; Schiele, Philipp; Schmelzer, Thomas. *Markowitz Portfolio Construction at Seventy*. 2024.  
  https://arxiv.org/abs/2401.05080
- Cai, Yongyang; Judd, Kenneth; Xu, Rong. *Numerical Solution of Dynamic Portfolio Optimization with Transaction Costs*. 2020.  
  https://arxiv.org/abs/2003.01809
- Chen, Pierre; Lezmi, Edmond; Roncalli, Thierry; Xu, Jiali. *A Note on Portfolio Optimization with Quadratic Transaction Costs*. 2020.  
  https://arxiv.org/abs/2001.01612
- Ekren, Ibrahim; Liu, Ren; Muhle-Karbe, Johannes. *Optimal Rebalancing Frequencies for Multidimensional Portfolios*. 2015/2017.  
  https://arxiv.org/abs/1510.05097
- Fan, Qingliang; Medeiros, Marcelo C.; Yang, Hanming; Yang, Songshan. *Cost-aware Portfolios in a Large Universe of Assets*. 2024/2025.  
  https://arxiv.org/abs/2412.11575
- Hautsch, Nikolaus; Voigt, Stefan. *Large-Scale Portfolio Allocation Under Transaction Costs and Model Uncertainty*. 2019.  
  https://arxiv.org/abs/1709.06296
- Kallsen, Jan; Li, Shen. *Portfolio Optimization under Small Transaction Costs: a Convex Duality Approach*. 2013.  
  https://arxiv.org/abs/1309.3479
- Liu, Ren; Muhle-Karbe, Johannes; Weber, Marko H. *Rebalancing with Linear and Quadratic Costs*. 2014/2017.  
  https://arxiv.org/abs/1402.5306
