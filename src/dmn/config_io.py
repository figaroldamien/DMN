from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import RunConfig


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".toml":
        import tomllib

        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError("Unsupported config format. Use .json or .toml")


def load_run_config(path: str | Path) -> RunConfig:
    p = Path(path)
    raw = _read_mapping(p)

    cfg = RunConfig()

    backtest_raw = raw.get("backtest", {}) if isinstance(raw.get("backtest", {}), dict) else {}
    flat_backtest_keys = {
        "sigma_target_annual",
        "vol_span",
        "cost_bps",
        "portfolio_vol_target",
        "min_obs",
    }

    top_level_updates = {
        k: raw[k]
        for k in ("market", "ticker", "start", "sector", "sub_sector", "run_ml", "run_dmn")
        if k in raw
    }
    if top_level_updates:
        cfg = replace(cfg, **top_level_updates)

    backtest_updates = {k: backtest_raw[k] for k in flat_backtest_keys if k in backtest_raw}
    backtest_updates.update({k: raw[k] for k in flat_backtest_keys if k in raw})
    if backtest_updates:
        cfg.backtest = replace(cfg.backtest, **backtest_updates)

    return cfg


def merge_cli_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    overrides = {}
    for k in ("market", "ticker", "start", "sector", "sub_sector", "run_ml", "run_dmn"):
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v
    if overrides:
        cfg = replace(cfg, **overrides)

    bt_overrides = {}
    for k in ("sigma_target_annual", "vol_span", "cost_bps", "portfolio_vol_target", "min_obs"):
        v = getattr(args, k, None)
        if v is not None:
            bt_overrides[k] = v
    if bt_overrides:
        cfg.backtest = replace(cfg.backtest, **bt_overrides)

    return cfg
