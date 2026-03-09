# Monthly Data Checklist

Use this checklist when updating `src/market_tickers_data/data`.

## 1. Run dry-run

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --dry-run
```

## 2. Refresh constituents (optional)

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --refresh nasdaq100 cac40 --dry-run
```

If output looks correct:

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --refresh nasdaq100 cac40
```

## 3. Validate project behavior

Run at least one quick command:

```bash
python3 -m dmn.cli --market cac40 --start 2020-01-01 --no-print-config
```

## 4. Commit with traceability

Include:
- Month/year (example: `2026-03`)
- Sources used (`Wikipedia Nasdaq-100`, `Wikipedia CAC 40`)
- Count of `added/removed/changed` tickers

## 5. Troubleshooting

- If web refresh fails, run normalize-only mode (no `--refresh`) and update files manually.
- If a CAC symbol mapping looks wrong (`.PA` vs `.AS` / `.MI`), keep the ticker that matches Yahoo Finance in your existing dataset.
- If sectors are blank for new entries, set temporary values and complete manually before merge.
