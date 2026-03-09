---
name: monthly-data-refresh
description: Update and validate monthly market component files under `src/market_tickers_data/data` (NASDAQ100, CAC40, index, dataset). Use when the user asks to refresh constituents, normalize JSON files, review additions/removals, or run a recurring monthly data maintenance workflow.
---

# Monthly Data Refresh

Run a repeatable monthly workflow to refresh constituent files in `src/market_tickers_data/data`.
Use the bundled script to fetch public constituents for NASDAQ100/CAC40, preserve local metadata (sector/sub_sector/category), normalize all JSON files, and print a diff summary.

## Workflow

1. Inspect current files in `src/market_tickers_data/data/*.json`.
2. Run the updater script in dry-run mode first.
3. Review `added` / `removed` / `changed` tickers in the terminal output.
4. Re-run without dry-run to write changes.
5. Run a quick smoke backtest command to ensure no broken ticker format.
6. Commit with a message that includes the month and data source.

## Commands

Preview only (recommended first):

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --dry-run
```

Refresh NASDAQ100 + CAC40 from Wikipedia, then normalize all files:

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --refresh nasdaq100 cac40
```

Normalize only (no network):

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py
```

## Rules

- Keep output schema consistent with existing files:
  - Required keys: `ticker`, `category`, `description`
  - Optional keys: `sector`, `sub_sector`
- Keep ticker format compatible with Yahoo Finance (e.g. `AIR.PA`, `MT.AS`, `STLAM.MI`).
- Preserve existing sector/sub-sector metadata when source tables do not provide them.
- Prefer dry-run before writing.

## References

- Monthly checklist and troubleshooting:
  [references/monthly-checklist.md](references/monthly-checklist.md)
