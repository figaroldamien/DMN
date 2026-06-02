# `optimal_tf` User Manual

Last updated: 2026-05-27

## Purpose

This document is the living user manual for `optimal_tf`.

It should be updated:
- at the end of each session that changes behavior, CLI, config, or outputs,
- or whenever explicitly requested.

See also:
- [optimal_tf_specifications.md](/Users/damien.figarol/trading_app_lab/docs/optimal_tf_specifications.md) for the functional scope,
- [optimal_tf_architecture.md](/Users/damien.figarol/trading_app_lab/docs/optimal_tf_architecture.md) for design decisions and module layout,
- [optimal_tf_dashboard.md](/Users/damien.figarol/trading_app_lab/docs/optimal_tf_dashboard.md) for the local Streamlit UI,
- [optimal_tf_strategies.md](/Users/damien.figarol/trading_app_lab/docs/optimal_tf_strategies.md) for strategy descriptions.

## Project Location

Repository root:
- `/Users/damien.figarol/trading_app_lab`

Package location:
- `src/optimal_tf`

Shared infrastructure location:
- `src/trading_core`

## Installation

From the repository root:

```bash
cd /Users/damien.figarol/trading_app_lab
.venv/bin/python -m pip install --no-build-isolation -e .
```

Why `--no-build-isolation`:
- in this environment, pip build isolation may try to resolve build dependencies over the network,
- the local editable install works reliably without that extra isolation step.

## Available CLI

Current CLIs:
- `optimal-tf`
- `optimal-tf-evaluate`
- `optimal-tf-compare`

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

Implementation note:
- `optimal_tf` is now primarily the application layer,
- shared mechanics for market data, feature primitives, covariance cleaning, periodic evaluation, comparison runs, and reporting exports mostly live in `trading_core`.

Workflow:
1. load the config,
2. resolve the universe,
3. download price history with `yfinance`,
4. resolve the effective allocation date,
5. compute the strategy state directly for that date,
6. print the weights and execution time,
7. optionally export them.

The evaluation CLI runs a periodic backtest:
1. load the config,
2. resolve the universe,
3. download price history,
4. generate rebalance dates from the requested frequency,
5. build the covariance cache once for the run,
6. compute portfolio weights at each rebalance date,
7. apply the weights on the following holding period,
8. account for turnover and transaction costs,
9. optionally apply portfolio-level volatility targeting,
10. print the performance summary and execution time,
11. optionally export detailed results.

The comparison CLI runs several strategies on the same evaluation setup:
1. load the config,
2. resolve the universe,
3. download price history once,
4. run periodic evaluation for each requested strategy,
5. export one subdirectory per strategy,
6. export comparison tables and plots,
7. print a summary table and execution time.

Implementation note:
- during one evaluation run, the engine reuses a covariance cache across rebalance dates,
- `ToRP2` and `ToRP3` also reuse a cached `RP` factor context across rebalance dates.

When plot export is enabled, the exported chart overlays:
- the `optimal_tf` portfolio,
- a universe equal-weight rebalanced benchmark,
- an equal-weight buy-and-hold benchmark.

## Main Commands

### 1. Run with the example config

```bash
cd /Users/damien.figarol/trading_app_lab
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
- `LLTF`
- `ToRP0`
- `ToRP1`
- `ToRP2`
- `ToRP3`

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

You can also set default export paths in the config under `[output]`:

```toml
[output]
allocation_csv = "output/optimal_tf/weights.csv"
allocation_json = "output/optimal_tf/weights.json"
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

Equivalent config-driven defaults:

```toml
[output]
evaluation_dir = "output/optimal_tf/evaluation_run"
evaluation_plot = true
```

This export now includes:
- tabular CSV/JSON outputs,
- `performance.png` with the portfolio and two universe benchmarks.

Current evaluation export files:
- `weights_by_rebalance.csv`
- `base_weights_by_rebalance.csv`
- `effective_weights_by_rebalance.csv`
- `signal_scale.csv`
- `portfolio_vol_scale.csv`
- `daily_returns_gross.csv`
- `daily_returns_net.csv`
- `turnover.csv`
- `costs.csv`
- `summary.json`
- `performance.png` when plot export is enabled

### 11. Disable chart generation

```bash
.venv/bin/optimal-tf-evaluate \
  --config configs/optimal_tf.example.toml \
  --output-dir output/optimal_tf/evaluation_run \
  --no-output-plot
```

### 12. Compare several strategies

```bash
.venv/bin/optimal-tf-compare \
  --config configs/optimal_tf.example.toml \
  --strategies RP,ARP,ToRP0,ToRP1,ToRP2,ToRP3 \
  --output-dir output/optimal_tf/compare_run
```

By default, `optimal-tf-compare` cleans `--output-dir` before writing results.
Use `--no-clean-output-dir` to keep existing files.
You can also set the default strategy list in the config:

```toml
[compare]
strategies = ["RP", "ARP", "ToRP0", "ToRP1", "ToRP2", "ToRP3"]
```

If `--strategies` is omitted, `optimal-tf-compare` now falls back to `[compare].strategies`.
If `[compare].strategies` is also absent, it falls back to the single strategy from `[evaluation]` / `[allocation]`.

You can also set comparison output defaults in the config:

```toml
[output]
compare_dir = "output/optimal_tf/compare_run"
compare_clean_dir = true
compare_plot = true
```

## Config Output Section

The TOML config can now carry output defaults for all three CLIs:

```toml
[output]
allocation_csv = "output/optimal_tf/weights.csv"
allocation_json = "output/optimal_tf/weights.json"
evaluation_dir = "output/optimal_tf/evaluation_run"
evaluation_plot = true
compare_dir = "output/optimal_tf/compare_run"
compare_clean_dir = true
compare_plot = true
```

Precedence rule:
- CLI flags override config values.
- If no CLI output flag is provided, the command falls back to `[output]`.
- `optimal-tf-compare` still requires an output directory overall, but it can now come from `--output-dir` or `[output].compare_dir`.

## Config Compare Section

The TOML config can also define the default strategy set for `optimal-tf-compare`:

```toml
[compare]
strategies = ["RP", "ARP", "LLTF", "ToRP0", "ToRP1", "ToRP2", "ToRP3"]
```

## Output Format

Standard output prints:
- strategy name,
- universe name,
- requested date,
- effective allocation date,
- execution time in seconds,
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
- `signal_scale`
- `base_weights`
- `weights`

Evaluation export notes:
- `daily_returns_gross.csv` contains portfolio returns before transaction costs,
- `daily_returns_net.csv` contains portfolio returns after transaction costs,
- `weights_by_rebalance.csv` remains the effective rebalance exposure used in the backtest,
- `base_weights_by_rebalance.csv` stores the structural portfolio before signal amplitude,
- `effective_weights_by_rebalance.csv` stores the exposure after `signal_scale`,
- `portfolio_vol_scale.csv` stores the separate portfolio-level volatility-targeting overlay.

Comparison export notes:
- `manifest.json` describes the compared strategies and available views,
- `inputs.json` stores the effective run configuration,
- `strategies/<strategy>/...` reuses the same per-strategy export contract as `optimal-tf-evaluate`,
- `comparison/summary_table.csv` stores one row per strategy,
- `comparison/nav_comparison.csv` stores cumulative NAV series side by side,
- `comparison/drawdown_comparison.csv` stores drawdown series side by side,
- `comparison/plots/` stores the first comparison PNG charts.

CLI note:
- both `optimal-tf` and `optimal-tf-evaluate` now print `execution_time_seconds` in their text output.

Architecture note:
- although the public commands remain under `optimal_tf`, the shared execution engine and export helpers now live mainly in `trading_core.backtest` and `trading_core.reporting`.

## Configuration File

Example config:
- [optimal_tf.example.toml](/Users/damien.figarol/trading_app_lab/configs/optimal_tf.example.toml)

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

- `covariance_window`
  Main lookback window used for covariance estimation before matrix cleaning.
  The standard covariance path now uses a fixed-window estimate rather than EWMA covariance smoothing.

- `corr_span`
  Legacy compatibility alias for the covariance window.
  It is no longer the primary documented parameter.

- `covariance_alpha`
  Compatibility parameter for covariance estimation, and still the direct smoothing parameter used internally by `LLTF`.
  On the standard covariance path, it is only used as a fallback to derive an effective window when `covariance_window` is not provided.

- `covariance_min_periods`
  Minimum observations required before covariance estimation starts producing matrices.
  Current default: `252`
  Constraint:
  `covariance_min_periods <= covariance_window`

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
  - `rie_reference`

- `linear_shrinkage`
  Shrinkage intensity used when `cleaning_method = "linear_shrinkage"`.

- `rie_bandwidth`
  Bandwidth used by the native `rie` cleaner.

- `rie_reference`
  Optional benchmark-only cleaner backed by the external `rie-estimator` package.
  It is not required for normal runs and is mainly useful to compare the native implementation against a reference implementation on the same data pipeline.

- `trend_alpha`
  Exponential smoothing coefficient for trend-following signals.

- `trend_span`
  Deprecated compatibility parameter for trend smoothing.
  It is converted to an EWMA coefficient when `trend_alpha` is not provided.

- `torp_signal_gain`
  Additional multiplicative gain applied only to `ToRP3` after factor-trend normalization.
  It is used to calibrate the strength of `signal_scale` without changing the other strategies.

- `lltf_l2_reg`
  Ridge regularization applied to the empirical `LLTF` virtual-asset covariance before inversion.
  It is a numerical stabilizer for the lead-lag optimizer.

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
  - `LLTF`
  - `ToRP0`
  - `ToRP1`
  - `ToRP2`
  - `ToRP3`

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
  - `LLTF`
  - `ToRP0`
  - `ToRP1`
  - `ToRP2`
  - `ToRP3`

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
  The current engine applies weights after each rebalance date and charges transaction cost on the first following trading day.

## Current Defaults

The current example config uses:
- `universe.name = "world_index"`
- `start = "2000-01-01"`
- `vol_span = 60`
- `covariance_window = 252`
- `covariance_min_periods = 252`
- `cleaning_method = "rie"`
- `trend_alpha = 0.01`
- `torp_signal_gain = 5.0`
- `lltf_l2_reg = 0.0001`
- `sigma_target_annual = 0.15`
- `cost_bps = 25.0`
- `long_only = true`
- `allocation.strategy = "ARP"`
- `evaluation.strategy = "ARP"`
- `evaluation.rebalance_frequency = "monthly"`
- `evaluation.evaluation_start = "2015-01-01"`

## Current Limitations

- Real data currently comes from `yfinance`.
- `optimal_tf` still keeps a few thin compatibility facades so existing imports and tests continue to work during the refactor.
- The current volatility targeting implementation works at the portfolio return level, not yet through a leverage-aware position rescaling layer recorded in the exported weights.
- `ToRP0` is the original implementation: asset-by-asset trend overlay on top of `RP`.
- `LLTF` is an empirical cross-asset lead-lag trend-following strategy inspired by arXiv:1410.8409.
- `ToRP1` measures a common trend signal on the `RP` factor itself; this is closer to the reference paper, but remains a simplified implementation.
- `ToRP2` is the most article-aligned current variant: it uses the trend of the `RP` factor return stream itself and neutralizes FX in that factor when metadata is available.
- `ToRP3` preserves the amplitude of the factor signal explicitly through `signal_scale` and `effective_weights`.
- `ToRP3` now stores a volatility-normalized factor signal, scaled by `torp_signal_gain`, rather than a raw factor return trend.
- `ToRP2` and `ToRP3` now benefit from per-run `RP` factor caching in the evaluation engine, which materially improves execution time relative to the first date-centric implementation.
- The standard covariance estimator now uses a fixed window plus cleaning, which is simpler to audit than the previous `EWMA covariance + cleaning` combination.
- The current data-quality filter is intentionally simple and threshold-based.
- The current `RIE` is a first native implementation and still needs validation against an external reference.
- Export metadata is still minimal and does not yet capture the full effective run configuration.

## What Is Next

The main planned improvements are:
- validation/refinement of the native `RIE`,
- anomaly diagnostics export,
- more faithful `ToRP` variants,
- portfolio combinations from the note,
- richer evaluation reports and benchmarks.

Near-term validation plan:
- compare the native `RIE` output against an external reference,
- run evaluation sweeps for `empirical`, `linear_shrinkage`, and `rie`,
- inspect the effect of the cleaner on `ARP`, `NM`, `ToRP0`, `ToRP1`, `ToRP2`, and `ToRP3`.
- inspect `LLTF` robustness to regularization and universe size.

## Benchmarking Cleaners

A dedicated benchmark script is now available:

```bash
cd /Users/damien.figarol/trading_app_lab
PYTHONPATH=src .venv/bin/python -m optimal_tf.scripts.benchmark_cleaners \
  --config configs/optimal_tf.example.toml \
  --universe sp500 \
  --strategies RP,ARP,NM \
  --methods empirical,linear_shrinkage,rie,rie_reference
```

What it produces:
- `matrix_benchmark.csv`: matrix-level diagnostics on the same normalized-return sample used by the in-house pipeline
- `strategy_benchmark.csv`: final strategy metrics for each cleaning method and strategy
- `summary.json`: run metadata and file locations

If `rie-estimator` is installed, the script also adds a `rie_reference_pipe` row in the matrix benchmark to compare against the external package's own normalization path.

Install the optional benchmark dependency with:

```bash
.venv/bin/python -m pip install rie-estimator
```

Refactor status:
- the shared extraction to `trading_core` is largely complete for market data, features, risk, rebalance logic, periodic backtests, comparison runs, and reporting,
- `optimal_tf` now mainly owns strategies, config, and CLI entry points.

## Tests

To run the current `optimal_tf` test suite:

```bash
cd /Users/damien.figarol/trading_app_lab
.venv/bin/python -m unittest discover -s tests/optimal_tf -p 'test_*.py'
```

To run both the application tests and the shared-core tests:

```bash
cd /Users/damien.figarol/trading_app_lab
.venv/bin/python -m unittest discover -s tests/optimal_tf -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests/trading_core -p 'test_*.py'
```

## Troubleshooting

### `No module named optimal_tf`

Reinstall the project in editable mode:

```bash
cd /Users/damien.figarol/trading_app_lab
.venv/bin/python -m pip install --no-build-isolation -e .
```

### CLI works with `python -m` but not with `optimal-tf`

Re-run the editable install command above so the script entry point is recreated.

### `optimal-tf-evaluate` is not found

Re-run the editable install command above so the new script entry point is installed:

```bash
cd /Users/damien.figarol/trading_app_lab
.venv/bin/python -m pip install --no-build-isolation -e .
```

### No weights available for the requested date

Typical causes:
- the requested date is before enough history is available,
- there is not enough price data to satisfy `covariance_min_periods`,
- the config is incoherent because `covariance_min_periods > covariance_window`,
- or, in compatibility mode, there is not enough history for the legacy `corr_span` or `covariance_alpha` fallback,
- the downloaded universe is too sparse over the requested interval.
