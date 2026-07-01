---
name: monthly-data-refresh
description: Update and validate monthly market component files under `data/market_tickers/universes` (NASDAQ100, CAC40, EUROSTOXX50, EUROSTOXX600, SP500, DJI, SBF120, index, world_index, dataset). Use when the user asks to refresh constituents, normalize JSON files, review additions/removals, or run a recurring monthly data maintenance workflow.
---

# Monthly Data Refresh

Run a repeatable monthly workflow to refresh constituent files in `data/market_tickers/universes`.
Use the bundled script to fetch public constituents for NASDAQ100/CAC40/EUROSTOXX50/SP500/DJI, preserve local metadata (sector/sub_sector/category), apply project metadata overrides for known gaps, harmonize hierarchy labels against the project vocabulary, normalize all files, and print a diff plus completeness summary.

## Workflow

1. Inspect current files in `data/market_tickers/universes/*.json`.
2. Run the updater script in dry-run mode first.
3. Review `added` / `removed` / `changed` tickers in the terminal output.
4. Review the completeness summary for `sector` / `sub_sector`, especially after constituent additions.
5. Review hierarchy label consistency:
   - For French universes such as `sbf120`, prefer the local `cac40` vocabulary when a ticker exists in both files.
   - Normalize obvious naming variants (`Aerospace & Defense` vs `Aerospace and Defense`, `Diagnostics & Research` vs `Diagnostics and Research`, `REIT-Retail` vs `Retail REITs`, etc.).
6. Validate Yahoo compatibility for European universes:
   - For `eurostoxx600`, re-test suspicious symbols that return `possibly delisted` / `no timezone found`.
   - If a company is still listed but Yahoo now exposes another primary symbol, add it to the local replacement map.
   - If the company was acquired, delisted, or moved out of the accessible listing set, remove the dead symbol from the seed universe rather than keeping a permanently broken ticker.
7. Re-run without dry-run to write changes.
8. Run a quick smoke backtest command to ensure no broken ticker format.
9. Commit with a message that includes the month and data source.

## Commands

Preview only (recommended first):

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --dry-run
```

Refresh NASDAQ100 + CAC40 + EUROSTOXX50 from public sources, then normalize all files:

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py --refresh nasdaq100 cac40 eurostoxx50 sp500 dji
```

Normalize only (no network):

```bash
python3 skills/monthly-data-refresh/scripts/update_data_files.py
```

If Python hits SSL certificate issues while fetching public pages, prefer the project venv plus the bundled `certifi` CA file:

```bash
SSL_CERT_FILE=.venv/lib/python3.14/site-packages/certifi/cacert.pem \
  .venv/bin/python skills/monthly-data-refresh/scripts/update_data_files.py --refresh nasdaq100 cac40 eurostoxx50 sp500 dji --dry-run
```

## Rules

- Keep output schema consistent with existing files:
  - Required keys: `ticker`, `category`, `description`
  - Optional keys: `sector`, `sub_sector`
- Keep ticker format compatible with Yahoo Finance (e.g. `AIR.PA`, `MT.AS`, `STLAM.MI`).
- Preserve existing sector/sub-sector metadata when source tables do not provide them.
- Apply project-level metadata overrides to fill known gaps after refresh.
- Harmonize sector/sub-sector labels after refresh so files do not mix multiple naming vocabularies.
- Maintain explicit Yahoo symbol replacement/removal maps for unstable European universes when corporate actions make old symbols unusable.
- Review completeness output and add new overrides when fresh constituents arrive without classification.
- Prefer dry-run before writing.
- If a public constituent table exposes issuer names or local exchange mnemonics that do not directly match Yahoo tickers, use `yfinance.Search(...)` as a fallback resolver and prefer the primary local listing (`.PA`, `.AS`, `.MI`, `.DE`, `.SW`, `.L`, etc.) over ADR/OTC variants.
- If `yfinance.Search(...)` returns homonyms or the wrong issuer, query `yfinance.Ticker(ticker).info` on the intended ticker to fetch the exact `sector` / `industry` metadata instead of trusting a fuzzy name match.

## References

- Monthly checklist and troubleshooting:
  [references/monthly-checklist.md](references/monthly-checklist.md)
