from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import OptimizationConfig, RunConfig


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
    model_raw = raw.get("model", {}) if isinstance(raw.get("model", {}), dict) else {}
    optimization_raw = raw.get("optimization", {}) if isinstance(raw.get("optimization", {}), dict) else {}
    flat_backtest_keys = {
        "sigma_target_annual",
        "vol_span",
        "cost_bps",
        "portfolio_vol_target",
        "min_obs",
    }
    model_keys = {
        "hidden",
        "dropout",
        "use_ticker_embedding",
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

    model_updates = {k: model_raw[k] for k in model_keys if k in model_raw}
    legacy_model_map = {
        "dmn_hidden": "hidden",
        "dmn_dropout": "dropout",
        "dmn_use_ticker_embedding": "use_ticker_embedding",
    }
    for legacy_key, model_key in legacy_model_map.items():
        if legacy_key in raw and model_key not in model_updates:
            model_updates[model_key] = raw[legacy_key]
    if model_updates:
        cfg.model = replace(cfg.model, **model_updates)

    if optimization_raw:
        required_optimization_keys = (
            "strategy",
            "metric",
            "hidden_values",
            "dropout_values",
            "batch_size_values",
            "learning_rate_values",
            "epochs_values",
        )
        missing_keys = [key for key in required_optimization_keys if key not in optimization_raw]
        if missing_keys:
            raise ValueError(f"Missing optimization config keys: {missing_keys}")
        cfg.optimization = OptimizationConfig(
            strategy=optimization_raw["strategy"],
            metric=optimization_raw["metric"],
            hidden_values=list(optimization_raw["hidden_values"]),
            dropout_values=list(optimization_raw["dropout_values"]),
            batch_size_values=list(optimization_raw["batch_size_values"]),
            learning_rate_values=list(optimization_raw["learning_rate_values"]),
            epochs_values=list(optimization_raw["epochs_values"]),
        )

    return cfg


def merge_cli_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    overrides = {}
    for k in ("market", "ticker", "start", "sector", "sub_sector", "run_ml", "run_dmn"):
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v

    cli_market = getattr(args, "market", None)
    cli_ticker = getattr(args, "ticker", None)
    if cli_ticker is not None:
        overrides["market"] = None
        overrides["sector"] = None
        overrides["sub_sector"] = None
    elif cli_market is not None:
        overrides["ticker"] = None

    if overrides:
        cfg = replace(cfg, **overrides)

    bt_overrides = {}
    for k in ("sigma_target_annual", "vol_span", "cost_bps", "portfolio_vol_target", "min_obs"):
        v = getattr(args, k, None)
        if v is not None:
            bt_overrides[k] = v
    if bt_overrides:
        cfg.backtest = replace(cfg.backtest, **bt_overrides)

    model_overrides = {}
    cli_to_model = {
        "model_hidden": "hidden",
        "model_dropout": "dropout",
        "model_use_ticker_embedding": "use_ticker_embedding",
    }
    for cli_key, model_key in cli_to_model.items():
        v = getattr(args, cli_key, None)
        if v is not None:
            model_overrides[model_key] = v
    if model_overrides:
        cfg.model = replace(cfg.model, **model_overrides)

    return cfg


def merge_optimization_cli_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    optimization_keys = (
        "strategy",
        "metric",
        "hidden_values",
        "dropout_values",
        "batch_size_values",
        "learning_rate_values",
        "epochs_values",
    )
    optimization_overrides = {}
    for key in optimization_keys:
        value = getattr(args, key, None)
        if value is not None:
            optimization_overrides[key] = value
    if not optimization_overrides:
        return cfg

    base = cfg.optimization
    cfg.optimization = OptimizationConfig(
        strategy=optimization_overrides.get("strategy", base.strategy if base else ""),
        metric=optimization_overrides.get("metric", base.metric if base else ""),
        hidden_values=optimization_overrides.get("hidden_values", list(base.hidden_values) if base else []),
        dropout_values=optimization_overrides.get("dropout_values", list(base.dropout_values) if base else []),
        batch_size_values=optimization_overrides.get("batch_size_values", list(base.batch_size_values) if base else []),
        learning_rate_values=optimization_overrides.get(
            "learning_rate_values", list(base.learning_rate_values) if base else []
        ),
        epochs_values=optimization_overrides.get("epochs_values", list(base.epochs_values) if base else []),
    )
    return cfg
