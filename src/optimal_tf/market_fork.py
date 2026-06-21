from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketForkSnapshot:
    snapshot_version: int = 1
    created_at_utc: str = ""
    source_app: str = "optimal_tf_dashboard"
    source_service: str = ""
    label: str = ""
    config_path: str = ""
    market_universe: str = ""
    market_start: str | None = None
    market_as_of_date: str | None = None
    source_request: dict[str, Any] = field(default_factory=dict)
    source_context: dict[str, Any] = field(default_factory=dict)
    source_artifacts: dict[str, str] = field(default_factory=dict)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Series):
        return value.rename("value").reset_index().to_dict(orient="records")
    if isinstance(value, pd.DataFrame):
        return value.head(200).to_dict(orient="records")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_market_fork_snapshot(
    *,
    source_service: str,
    config_path: str,
    market_universe: str,
    market_start: str | None,
    market_as_of_date: str | None,
    source_request: Any,
    source_context: dict[str, Any],
    source_artifacts: dict[str, Path],
    label: str | None = None,
) -> MarketForkSnapshot:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    normalized_label = label or f"{source_service} {market_universe}".strip()
    return MarketForkSnapshot(
        created_at_utc=timestamp,
        source_service=source_service,
        label=normalized_label,
        config_path=config_path,
        market_universe=market_universe,
        market_start=market_start,
        market_as_of_date=market_as_of_date,
        source_request=_json_safe(source_request),
        source_context=_json_safe(source_context),
        source_artifacts={str(name): str(path) for name, path in source_artifacts.items()},
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "market-fork"


def write_market_fork_snapshot(snapshot: MarketForkSnapshot, output_dir: str | Path) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = snapshot.created_at_utc.replace(":", "").replace("-", "")
    filename = f"{timestamp}_{_slugify(snapshot.source_service)}_{_slugify(snapshot.market_universe)}.json"
    path = target_dir / filename
    path.write_text(json.dumps(asdict(snapshot), indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def load_market_fork_snapshot(path: str | Path) -> MarketForkSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MarketForkSnapshot(**payload)
