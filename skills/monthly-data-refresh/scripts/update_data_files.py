#!/usr/bin/env python3
"""Monthly refresh tool for src/market_tickers_data/data JSON files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


KEY_ORDER = ("ticker", "category", "sector", "sub_sector", "description")
DEFAULT_CATEGORY = "equity"


@dataclass(frozen=True)
class ChangeSummary:
    added: list[str]
    removed: list[str]
    changed: list[str]


def _canonical_row(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in KEY_ORDER:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
        if value == "":
            continue
        out[key] = value
    if "ticker" not in out:
        raise ValueError("Missing required key: ticker")
    if "category" not in out:
        out["category"] = DEFAULT_CATEGORY
    if "description" not in out:
        out["description"] = out["ticker"]
    return out


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}")
    return [_canonical_row(item) for item in raw if isinstance(item, dict)]


def canonicalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = _canonical_row(row)
        dedup[item["ticker"]] = item
    return [dedup[t] for t in sorted(dedup)]


def write_json_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")


def summarize_diff(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> ChangeSummary:
    old_map = {r["ticker"]: r for r in old}
    new_map = {r["ticker"]: r for r in new}

    old_keys = set(old_map)
    new_keys = set(new_map)

    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    changed = sorted(k for k in old_keys & new_keys if old_map[k] != new_map[k])
    return ChangeSummary(added=added, removed=removed, changed=changed)


def _pick_table_with_columns(url: str, expected_cols: tuple[str, ...]) -> pd.DataFrame:
    tables = pd.read_html(url)
    norm_cols = [tuple(str(c).strip().lower() for c in t.columns) for t in tables]
    for table, cols in zip(tables, norm_cols):
        if all(any(exp in col for col in cols) for exp in expected_cols):
            return table
    raise ValueError(f"No matching table on {url} for columns {expected_cols}")


def _normalize_symbol_for_cac(symbol: str, existing_by_base: dict[str, str]) -> str:
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    if symbol in existing_by_base:
        return existing_by_base[symbol]
    return f"{symbol}.PA"


def refresh_nasdaq100(existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    table = _pick_table_with_columns(url, ("ticker", "company"))
    cols = {str(c).strip().lower(): c for c in table.columns}
    ticker_col = cols[next(k for k in cols if "ticker" in k)]
    name_col = cols[next(k for k in cols if "company" in k)]

    existing_map = {r["ticker"]: r for r in existing}
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        ticker = str(row[ticker_col]).strip().upper()
        ticker = ticker.replace(".", "-")
        if not ticker or ticker == "NAN":
            continue
        base = existing_map.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "category": base.get("category", "equity"),
                "sector": base.get("sector"),
                "sub_sector": base.get("sub_sector"),
                "description": str(row[name_col]).strip(),
            }
        )
    return canonicalize_rows(rows)


def refresh_cac40(existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = "https://en.wikipedia.org/wiki/CAC_40"
    table = _pick_table_with_columns(url, ("ticker", "company"))
    cols = {str(c).strip().lower(): c for c in table.columns}
    ticker_col = cols[next(k for k in cols if "ticker" in k)]
    name_col = cols[next(k for k in cols if "company" in k)]

    existing_map = {r["ticker"]: r for r in existing}
    existing_by_base = {r["ticker"].split(".")[0].upper(): r["ticker"] for r in existing}

    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        symbol = str(row[ticker_col]).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        ticker = _normalize_symbol_for_cac(symbol, existing_by_base)
        base = existing_map.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "category": base.get("category", "equity"),
                "sector": base.get("sector"),
                "sub_sector": base.get("sub_sector"),
                "description": str(row[name_col]).strip(),
            }
        )
    return canonicalize_rows(rows)


def update_one_file(path: Path, new_rows: list[dict[str, Any]], dry_run: bool) -> ChangeSummary:
    old_rows = load_json_rows(path)
    new_rows = canonicalize_rows(new_rows)
    changes = summarize_diff(old_rows, new_rows)
    if not dry_run and (changes.added or changes.removed or changes.changed):
        write_json_rows(path, new_rows)
    return changes


def print_changes(label: str, changes: ChangeSummary) -> None:
    print(f"[{label}] added={len(changes.added)} removed={len(changes.removed)} changed={len(changes.changed)}")
    if changes.added:
        print(f"  + {', '.join(changes.added[:20])}")
    if changes.removed:
        print(f"  - {', '.join(changes.removed[:20])}")
    if changes.changed:
        print(f"  * {', '.join(changes.changed[:20])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monthly updater for market component JSON files.")
    parser.add_argument(
        "--data-dir",
        default="src/market_tickers_data/data",
        help="Directory containing *_components.json files.",
    )
    parser.add_argument(
        "--refresh",
        nargs="*",
        choices=["nasdaq100", "cac40"],
        default=[],
        help="Sources to refresh from public web tables.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    targets = [
        "dataset_components.json",
        "index_components.json",
        "nasdaq100_components.json",
        "cac40_components.json",
    ]

    for name in targets:
        path = data_dir / name
        if not path.exists():
            print(f"[skip] missing file: {path}")
            continue

        existing = load_json_rows(path)
        new_rows = canonicalize_rows(existing)

        if name == "nasdaq100_components.json" and "nasdaq100" in args.refresh:
            new_rows = refresh_nasdaq100(existing)
        elif name == "cac40_components.json" and "cac40" in args.refresh:
            new_rows = refresh_cac40(existing)

        changes = update_one_file(path, new_rows, dry_run=args.dry_run)
        print_changes(name, changes)

    if args.dry_run:
        print("Dry-run complete: no files were written.")
    else:
        print("Update complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
