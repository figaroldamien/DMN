from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_output_dir(path: str | Path | None, *, clean: bool = False) -> Path | None:
    if path is None:
        return None
    outdir = Path(path)
    if clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_request_json(outdir: Path | None, request: Any) -> Path | None:
    if outdir is None:
        return None
    path = outdir / 'request.json'
    path.write_text(json.dumps(_json_safe(request), indent=2), encoding='utf-8')
    return path


def write_json(outdir: Path | None, name: str, payload: dict[str, Any]) -> Path | None:
    if outdir is None:
        return None
    path = outdir / name
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding='utf-8')
    return path
