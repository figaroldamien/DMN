# `optimal_tf` User Manual

Last updated: 2026-03-27

## Purpose

This document is the living user manual for `optimal_tf`.

It should be updated:
- at the end of each session that changes behavior, CLI, config, or outputs,
- or whenever explicitly requested.

See also:
- [optimal_tf_specifications.md](/Users/damien.figarol/DMN/docs/optimal_tf_specifications.md) for the functional scope,
- [optimal_tf_architecture.md](/Users/damien.figarol/DMN/docs/optimal_tf_architecture.md) for design decisions and module layout.

## Project Location

Repository root:
- `/Users/damien.figarol/DMN`

Package location:
- `src/optimal_tf`

## Installation

From the repository root:

```bash
cd /Users/damien.figarol/DMN
.venv/bin/python -m pip install --no-build-isolation -e .
```

Why `--no-build-isolation`:
- in this environment, pip build isolation may try to resolve build dependencies over the network,
- the local editable install works reliably without that extra isolation step.

## Available CLI

Current CLIs:
- `optimal-tf`
- `optimal-tf-evaluate`

Equivalent module form:

```bash
.venv/bin/python -m optimal_tf.cli --help
```

Installed script form:

```bash
.venv/bin/optimal-tf --help
```

## What The CLI Does

The current CLI computes portfolio weights for a single allocation date.

Workflow:
1. load the config,
2. resolve the universe,
3. download price history with `yfinance`,
4. estimate weights using the chosen portfolio recipe,
5. resolve the allocation date,
6. print the weights,
7. optionally export them.

The evaluation CLI runs a periodic backtest:
1. load the config,
2. resolve the universe,
3. download price history,
4. generate rebalance dates from the requested frequency,
5. compute portfolio weights at each rebalance date,
6. apply the weights on the following holding period,
7. account for turnover and transaction costs,
8. optionally apply portfolio-level volatility targeting,
9. print the performance summary,
10. optionally export detailed results.

When plot export is enabled, the exported chart overlays:
- the `optimal_tf` portfolio,
- a universe equal-weight rebalanced benchmark,
- an equal-weight buy-and-hold benchmark.

## Main Commands

### 1. Run with the example config

```bash
cd /Users/damien.figarol/DMN
.venv/bin/optimal-tf --config configs/optimal_tf.example.toml
```

### 2. Override the universe

```bash
.venv/bin/optimal-tf --config configs/optimal_tf.example.toml --universe cac40
```

### 3. Compute weights at a specific date

```bash
.venv/bin/optimal-tf --config configs/optimal_tf.example.toml --date 2026-03-27
```

### 4. Switch strategy

```bash
.venv/bin/optimal-tf --config configs/optimal_tf.example.toml --strategy ARP
```

Available strategies today:
- `RP`
- `ARP`
- `NM`
- `EW`
- `ToRP`

### 5. Force long-only

```bash
.venv/bin/optimal-tf --config configs/optimal_tf.example.toml --long-only
```

### 6. Export weights

```bash
.venv/bin/optimal-tf \
  --config configs/optimal_tf.example.toml \
  --output-csv output/optimal_tf/weights.csv \
  --output-json output/optimal_tf/weights.json
```

### 7. Run a periodic evaluation

```bash
.venv/bin/optimal-tf-evaluate --config configs/optimal_tf.example.toml
```

### 8. Evaluate with a custom rebalance frequency

```bash
.venv/bin/optimal-tf-evaluate \
  --config configs/optimal_tf.example.toml \
  --rebalance-frequency weekly
```

### 9. Evaluate on a custom interval

```bash
.venv/bin/optimal-tf-evaluate \
  --config configs/optimal_tf.example.toml \
  --evaluation-start 2018-01-01 \
  --evaluation-end 2024-12-31
```

### 10. Export evaluation outputs

```bash
.venv/bin/optimal-tf-evaluate \
  --config configs/optimal_tf.example.toml \
  --output-dir output/optimal_tf/evaluation_run
```

This export now includes:
- tabular CSV/JSON outputs,
- `performance.png` with the portfolio and two universe benchmarks.

### 11. Disable chart generation

```bash
.venv/bin/optimal-tf-evaluate \
  --config configs/optimal_tf.example.toml \
  --output-dir output/optimal_tf/evaluation_run \
  --no-output-plot
```

## Output Format

Standard output prints:
- strategy name,
- universe name,
- requested date,
- effective allocation date,
- number of non-zero assets,
- weights sorted by value.

CSV export columns:
- `date`
- `strategy`
- `universe`
- `ticker`
- `weight`

JSON export structure:
- `date`
- `strategy`
- `universe`
- `weights`

## Configuration File

Example config:
- [optimal_tf.example.toml](/Users/damien.figarol/DMN/configs/optimal_tf.example.toml)

### `[universe]`

- `name`
  Universe name resolved through `market_tickers_data`.
  Example: `cac40`

- `start`
  Start date for downloading price history.
  Example: `2000-01-01`

### `[estimation]`

- `vol_span`
  EWMA span for volatility estimation.
  Current default: `60`

- `corr_span`
  Rolling window length for correlation estimation.
  Current default: `252`

- `corr_min_periods`
  Minimum observations required before a correlation matrix is produced.
  Current default: `252`

- `max_abs_return`
  Data-quality guardrail for returns.
  Any daily return with absolute value above this threshold is excluded before
  estimation, evaluation, and benchmark construction.
  Current default: `1.0`

- `cleaning_method`
  Matrix cleaning method.
  Supported today:
  - `empirical`
  - `linear_shrinkage`
  - `rie`

- `linear_shrinkage`
  Shrinkage intensity used when `cleaning_method = "linear_shrinkage"`.

- `rie_bandwidth`
  Reserved for the future RIE implementation.

- `trend_span`
  Reserved for future trend-driven portfolio recipes such as `ToRP`.

### `[backtest]`

This section currently controls portfolio-level conventions even for the single-date CLI.

- `sigma_target_annual`
  Annual target volatility parameter.
  Current default: `0.15`

- `portfolio_vol_target`
  Whether to apply portfolio-level volatility targeting.
  This now applies in the periodic evaluation engine.

- `portfolio_vol_span`
  EWMA span used for portfolio volatility targeting.
  In the periodic evaluation engine, the targeter uses a one-day lag to avoid look-ahead bias.

- `cost_bps`
  Transaction cost placeholder in basis points.
  Not heavily used by the current single-date allocation CLI yet.

- `long_only`
  If `true`, negative weights are clipped to zero and the result is renormalized.
  If `false`, long/short weights are allowed.

### `[allocation]`

- `strategy`
  Portfolio recipe used by the CLI.
  Current values:
  - `RP`
  - `ARP`
  - `NM`
  - `EW`
  - `ToRP`

- `date`
  Optional allocation date.
  If omitted, the CLI uses today's date and then resolves to the latest available market date on or before today.

### `[evaluation]`

- `strategy`
  Strategy used by the evaluation engine.
  Current values:
  - `RP`
  - `ARP`
  - `NM`
  - `EW`
  - `ToRP`

- `rebalance_frequency`
  Supported values:
  - `daily`
  - `weekly`
  - `monthly`
  - `quarterly`
  - `yearly`

- `evaluation_start`
  Optional start date for the backtest window.

- `evaluation_end`
  Optional end date for the backtest window.

## Current Defaults

The current example config uses:
- `universe.name = "cac40"`
- `start = "2000-01-01"`
- `vol_span = 60`
- `corr_span = 252`
- `corr_min_periods = 252`
- `cleaning_method = "empirical"`
- `trend_span = 252`
- `sigma_target_annual = 0.15`
- `long_only = false`
- `allocation.strategy = "RP"`
- `evaluation.strategy = "RP"`
- `evaluation.rebalance_frequency = "monthly"`
- `evaluation.evaluation_start = "2015-01-01"`

## Current Limitations

- Real data currently comes from `yfinance`.
- The current volatility targeting implementation works at the portfolio return level, not yet through a leverage-aware position rescaling layer recorded in the exported weights.
- `ToRP` is currently a first operational V1, not yet the final paper-faithful variant.
- The current data-quality filter is intentionally simple and threshold-based.
- The current `RIE` is a first native implementation and still needs validation against an external reference.

## What Is Next

The main planned improvements are:
- validation/refinement of the native `RIE`,
- anomaly diagnostics export,
- more faithful `ToRP`,
- portfolio combinations from the note,
- richer evaluation reports and benchmarks.

Near-term validation plan:
- compare the native `RIE` output against an external reference,
- run evaluation sweeps for `empirical`, `linear_shrinkage`, and `rie`,
- inspect the effect of the cleaner on `ARP`, `NM`, and `ToRP`.

## Tests

To run the current `optimal_tf` test suite:

```bash
cd /Users/damien.figarol/DMN
.venv/bin/python -m unittest discover -s tests/optimal_tf -p 'test_*.py'
```

## Troubleshooting

### `No module named optimal_tf`

Reinstall the project in editable mode:

```bash
cd /Users/damien.figarol/DMN
.venv/bin/python -m pip install --no-build-isolation -e .
```

### CLI works with `python -m` but not with `optimal-tf`

Re-run the editable install command above so the script entry point is recreated.

### `optimal-tf-evaluate` is not found

Re-run the editable install command above so the new script entry point is installed:

```bash
cd /Users/damien.figarol/DMN
.venv/bin/python -m pip install --no-build-isolation -e .
```

### No weights available for the requested date

Typical causes:
- the requested date is before enough history is available,
- there is not enough price data to satisfy `corr_min_periods`,
- the downloaded universe is too sparse over the requested interval.
