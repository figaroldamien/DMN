# Monthly Data Checklist

Use this checklist when updating `data/market_tickers/universes`, including `world_index_components.json`, `eurostoxx50_components.json`, `eurostoxx600_components.json`, `sbf120_components.json`, `sp500_components.json`, and `dji_components.json` when present.

## 1. Run dry-run

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --dry-run
```

## 2. Refresh constituents (optional)

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --refresh nasdaq100 cac40 eurostoxx50 sp500 dji sp500 dji --dry-run
```

If output looks correct:

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --refresh nasdaq100 cac40 eurostoxx50 sp500 dji
```

## 3. Validate project behavior

Run at least one quick command:

```bash
python3 -m dmn.cli --market cac40 --start 2020-01-01 --no-print-config
```

## 4. Harmonize hierarchy labels

- Review sector / sub-sector vocabulary for mixed naming styles after every refresh or manual universe build.
- For `sbf120`, prefer the local `cac40` hierarchy when the same ticker exists in both universes.
- Normalize obvious textual variants before merge:
  - `Aerospace & Defense` -> `Aerospace and Defense`
  - `Diagnostics & Research` -> `Diagnostics and Research`
  - `Information Technology Services` -> `IT Services`
  - `REIT-Retail` / `REIT - Retail` -> `Retail REITs`
  - `Software-Application` / `Software-Infrastructure` -> `Software`

## 5. Validate Yahoo compatibility

- Re-test suspicious European tickers that fail with `possibly delisted`, `Quote not found`, or `no timezone found`.
- If Yahoo now exposes a different active symbol for the same issuer, add a deterministic replacement to the maintenance script.
- If the issuer was acquired, delisted, or renamed out of the old listing line, remove the dead symbol from the seed universe instead of keeping a permanently broken ticker.

## 6. Commit with traceability

Include:
- Month/year (example: `2026-03`)
- Sources used (`Wikipedia Nasdaq-100`, `Wikipedia CAC 40`, `Wikipedia EURO STOXX 50`, `Wikipedia S&P 500`, `Wikipedia Dow Jones Industrial Average`)
- Count of `added/removed/changed` tickers

## 7. Troubleshooting

- If web refresh fails, run normalize-only mode (no `--refresh`) and update files manually.
- If HTTPS fetches fail with `CERTIFICATE_VERIFY_FAILED`, rerun from `.venv/bin/python` and point `SSL_CERT_FILE` to `.venv/lib/python3.14/site-packages/certifi/cacert.pem`.
- If a CAC symbol mapping looks wrong (`.PA` vs `.AS` / `.MI`), keep the ticker that matches Yahoo Finance in your existing dataset.
- If a source exposes company names or exchange-specific mnemonics instead of Yahoo-ready tickers, probe candidates with `yfinance.Search(company_name)` and keep the symbol for the main local listing.
- If a company name is ambiguous (`SEB`, `SES`, etc.), validate the final symbol with `yfinance.Ticker(symbol).info` and use that exact ticker's metadata for `sector` / `sub_sector`.
- If sectors are blank for new entries, set temporary values and complete manually before merge.
