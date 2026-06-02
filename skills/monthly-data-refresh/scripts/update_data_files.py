#!/usr/bin/env python3
"""Monthly refresh tool for data/market_tickers/universes JSON files."""

from __future__ import annotations

import argparse
from io import StringIO
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


KEY_ORDER = ("ticker", "category", "sector", "sub_sector", "description")
DEFAULT_CATEGORY = "equity"

# Project-level metadata overrides used when public constituent sources do not
# provide sector hierarchies for newly added names.
METADATA_OVERRIDES: dict[str, dict[str, dict[str, str]]] = {
    "nasdaq100": {
        "LITE": {
            "sector": "Technology",
            "sub_sector": "Communication Equipment",
        },
        "SNDK": {
            "sector": "Technology",
            "sub_sector": "Computer Hardware",
        },
    },
    "cac40": {
        "BVI.PA": {
            "sector": "Industrials",
            "sub_sector": "Professional Services",
        },
        "EDEN.PA": {
            "sector": "Financials",
            "sub_sector": "Transaction and Payment Processing Services",
        },
        "STLAP.PA": {
            "sector": "Consumer Discretionary",
            "sub_sector": "Auto Manufacturers",
        },
        "URW.PA": {
            "sector": "Real Estate",
            "sub_sector": "Retail REITs",
        },
    },
}


@dataclass(frozen=True)
class ChangeSummary:
    added: list[str]
    removed: list[str]
    changed: list[str]


@dataclass(frozen=True)
class CompletenessSummary:
    total: int
    sector_filled: int
    sub_sector_filled: int
    both_filled: int
    missing_examples: list[str]


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
    if isinstance(raw, list):
        component_rows = raw
    elif isinstance(raw, dict) and isinstance(raw.get("components"), list):
        component_rows = raw["components"]
    else:
        raise ValueError(f"Expected list or object-with-components in {path}")
    return [_canonical_row(item) for item in component_rows if isinstance(item, dict)]


def _load_file_metadata(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if k != "components"}
    return {}


def canonicalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = _canonical_row(row)
        dedup[item["ticker"]] = item
    return [dedup[t] for t in sorted(dedup)]


def apply_metadata_overrides(universe: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overrides = METADATA_OVERRIDES.get(universe, {})
    if not overrides:
        return rows
    patched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        override = overrides.get(item.get("ticker", ""))
        if override:
            for key in ("sector", "sub_sector"):
                if not str(item.get(key, "") or "").strip() and key in override:
                    item[key] = override[key]
        patched.append(item)
    return patched


def summarize_completeness(rows: list[dict[str, Any]]) -> CompletenessSummary:
    missing_examples: list[str] = []
    sector_filled = 0
    sub_sector_filled = 0
    both_filled = 0
    for row in rows:
        sector = str(row.get("sector", "") or "").strip()
        sub_sector = str(row.get("sub_sector", "") or "").strip()
        if sector:
            sector_filled += 1
        if sub_sector:
            sub_sector_filled += 1
        if sector and sub_sector:
            both_filled += 1
        elif len(missing_examples) < 10:
            missing_examples.append(str(row.get("ticker", "")))
    return CompletenessSummary(
        total=len(rows),
        sector_filled=sector_filled,
        sub_sector_filled=sub_sector_filled,
        both_filled=both_filled,
        missing_examples=missing_examples,
    )


def write_json_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    metadata = _load_file_metadata(path) if path.exists() else {}
    if metadata:
        payload_obj = dict(metadata)
        payload_obj["components"] = rows
        payload = json.dumps(payload_obj, ensure_ascii=False, indent=2) + "\n"
    else:
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
    # Fetch the page ourselves so we can send a browser-like User-Agent.
    # Direct pd.read_html(url) calls are increasingly blocked by public sites.
    html = _fetch_text(url)
    tables = pd.read_html(StringIO(html))
    norm_cols = [tuple(str(c).strip().lower() for c in t.columns) for t in tables]
    for table, cols in zip(tables, norm_cols):
        if all(any(exp in col for col in cols) for exp in expected_cols):
            return table
    raise ValueError(f"No matching table on {url} for columns {expected_cols}")


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _clean_wiki_markup(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text)
    text = re.sub(r"<ref[^/]*/>", "", text)
    text = re.sub(r"\{\{Flagicon\|[^{}]+\}\}", "", text)
    text = re.sub(r"\{\{(?:FWB|BMAD|ISE|Euronext|EuronextParis|euronextParis)\|([^|{}=]+)(?:\|[^{}]*)*\}\}", r"\1", text)
    text = re.sub(r"\{\{[A-Z]{3}\}\}", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = text.replace("''", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_symbol_for_cac(symbol: str, existing_by_base: dict[str, str]) -> str:
    symbol = symbol.strip().upper()
    if "." in symbol:
        return symbol
    if symbol in existing_by_base:
        return existing_by_base[symbol]
    return f"{symbol}.PA"


def _normalize_symbol_with_existing(symbol: str, existing_map: dict[str, dict[str, Any]]) -> str:
    symbol = symbol.strip().upper()
    if symbol in existing_map:
        return symbol
    dashed = symbol.replace(".", "-")
    if dashed in existing_map:
        return dashed
    dotted = symbol.replace("-", ".")
    if dotted in existing_map:
        return dotted
    return dashed


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


def refresh_eurostoxx50(existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = "https://en.wikipedia.org/w/index.php?title=EURO_STOXX_50&action=raw"
    raw = _fetch_text(url)
    section = raw.split('id="constituents"', 1)[1]
    table = section.split("|}", 1)[0]

    sector_map = {
        "Health Care": "Healthcare",
        "Information Technology": "Technology",
        "Communication": "Communication Services",
    }
    existing_map = {r["ticker"]: r for r in existing}
    rows: list[dict[str, Any]] = []

    for block in table.split("|-"):
        block = block.strip()
        if not block.startswith("|"):
            continue

        if "||" in block:
            normalized = " ".join(line.strip() for line in block.splitlines())
            normalized = normalized.lstrip("|").strip()
            cells = [_clean_wiki_markup(cell) for cell in normalized.split("||")]
        else:
            cells = [_clean_wiki_markup(line.strip()[1:]) for line in block.splitlines() if line.strip().startswith("|")]

        if len(cells) < 6:
            continue

        ticker, _listing, name, _corp, _office, sector = cells[:6]
        base = existing_map.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "category": base.get("category", "equity"),
                "sector": base.get("sector", sector_map.get(sector, sector)),
                "sub_sector": base.get("sub_sector"),
                "description": name,
            }
        )
    return canonicalize_rows(rows)


def refresh_sp500(existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = _pick_table_with_columns(url, ("symbol", "security", "gics sector", "gics sub-industry"))
    cols = {str(c).strip().lower(): c for c in table.columns}
    ticker_col = cols[next(k for k in cols if "symbol" in k)]
    name_col = cols[next(k for k in cols if "security" in k)]
    sector_col = cols[next(k for k in cols if "gics sector" in k)]
    sub_sector_col = cols[next(k for k in cols if "gics sub-industry" in k)]

    existing_map = {r["ticker"]: r for r in existing}
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        raw_symbol = str(row[ticker_col]).strip().upper()
        if not raw_symbol or raw_symbol == "NAN":
            continue
        ticker = _normalize_symbol_with_existing(raw_symbol, existing_map)
        base = existing_map.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "category": base.get("category", "equity"),
                "sector": str(row[sector_col]).strip() or base.get("sector"),
                "sub_sector": str(row[sub_sector_col]).strip() or base.get("sub_sector"),
                "description": str(row[name_col]).strip(),
            }
        )
    return canonicalize_rows(rows)


def refresh_dji(existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    table = _pick_table_with_columns(url, ("company", "symbol", "sector"))
    cols = {str(c).strip().lower(): c for c in table.columns}
    ticker_col = cols[next(k for k in cols if "symbol" in k)]
    name_col = cols[next(k for k in cols if "company" in k)]
    industry_col = cols[next(k for k in cols if "sector" in k)]

    existing_map = {r["ticker"]: r for r in existing}
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        raw_symbol = str(row[ticker_col]).strip().upper()
        if not raw_symbol or raw_symbol == "NAN":
            continue
        ticker = _normalize_symbol_with_existing(raw_symbol, existing_map)
        base = existing_map.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "category": base.get("category", "equity"),
                "sector": base.get("sector", str(row[industry_col]).strip()),
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


def print_completeness(label: str, summary: CompletenessSummary) -> None:
    if summary.total == 0:
        print(f"  completeness: empty")
        return
    sector_pct = 100.0 * summary.sector_filled / summary.total
    sub_sector_pct = 100.0 * summary.sub_sector_filled / summary.total
    both_pct = 100.0 * summary.both_filled / summary.total
    print(
        "  completeness: "
        f"sector={summary.sector_filled}/{summary.total} ({sector_pct:.1f}%), "
        f"sub_sector={summary.sub_sector_filled}/{summary.total} ({sub_sector_pct:.1f}%), "
        f"both={summary.both_filled}/{summary.total} ({both_pct:.1f}%)"
    )
    if summary.both_filled < summary.total and summary.missing_examples:
        print(f"  missing examples: {', '.join(summary.missing_examples)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monthly updater for market component JSON files.")
    parser.add_argument(
        "--data-dir",
        default="data/market_tickers/universes",
        help="Directory containing universe *_components.json files.",
    )
    parser.add_argument(
        "--refresh",
        nargs="*",
        choices=["nasdaq100", "cac40", "eurostoxx50", "sp500", "dji"],
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
        "world_index_components.json",
        "nasdaq100_components.json",
        "cac40_components.json",
        "eurostoxx50_components.json",
        "sp500_components.json",
        "dji_components.json",
    ]

    for name in targets:
        path = data_dir / name
        if not path.exists():
            print(f"[skip] missing file: {path}")
            continue

        existing = load_json_rows(path)
        new_rows = canonicalize_rows(existing)
        universe_name = name.replace('_components.json', '')

        if name == "nasdaq100_components.json" and "nasdaq100" in args.refresh:
            new_rows = refresh_nasdaq100(existing)
        elif name == "cac40_components.json" and "cac40" in args.refresh:
            new_rows = refresh_cac40(existing)
        elif name == "eurostoxx50_components.json" and "eurostoxx50" in args.refresh:
            new_rows = refresh_eurostoxx50(existing)
        elif name == "sp500_components.json" and "sp500" in args.refresh:
            new_rows = refresh_sp500(existing)
        elif name == "dji_components.json" and "dji" in args.refresh:
            new_rows = refresh_dji(existing)

        new_rows = canonicalize_rows(apply_metadata_overrides(universe_name, new_rows))
        changes = update_one_file(path, new_rows, dry_run=args.dry_run)
        print_changes(name, changes)
        print_completeness(name, summarize_completeness(new_rows))

    if args.dry_run:
        print("Dry-run complete: no files were written.")
    else:
        print("Update complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
