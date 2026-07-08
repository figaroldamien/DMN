from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DashboardRunSummary:
    mode: str
    service: str
    universe: str | None
    request_type: str
    primary_subject: str | None
    strategy_count: int
    strategies: tuple[str, ...]
    cleaning_method: str | None
    covariance_window: int | None
    rebalance_frequency: str | None
    weight_smoothing_alpha: float | None
    evaluation_start: str | None
    evaluation_end: str | None
    output_dir: str | None
    artifact_root: str | None
    artifact_count: int
    artifact_names: tuple[str, ...]
    highlights_count: int
    warning_count: int


def build_dashboard_run_summary(
    *,
    mode: str,
    service: str,
    request: Any,
    universe: str | None,
    artifacts: Any = None,
    resolved: dict[str, Any] | None = None,
    highlights: dict[str, Any] | None = None,
    warnings: list[str] | tuple[str, ...] | None = None,
) -> DashboardRunSummary:
    payload = _request_payload(request)
    resolved_payload = dict(resolved or {})
    strategies = _resolve_strategies(payload, resolved_payload)
    primary_subject = _resolve_primary_subject(service=service, strategies=strategies, resolved=resolved_payload)
    artifact_root, artifact_names = _resolve_artifacts(artifacts)
    return DashboardRunSummary(
        mode=mode,
        service=service,
        universe=universe,
        request_type=type(request).__name__ if request is not None else "UnknownRequest",
        primary_subject=primary_subject,
        strategy_count=len(strategies),
        strategies=tuple(strategies),
        cleaning_method=_resolve_cleaning_method(payload, resolved_payload),
        covariance_window=_resolve_covariance_window(payload, resolved_payload),
        rebalance_frequency=_resolve_rebalance_frequency(payload, resolved_payload),
        weight_smoothing_alpha=_optional_float(
            resolved_payload.get("weight_smoothing_alpha", payload.get("weight_smoothing_alpha"))
        ),
        evaluation_start=_optional_str(payload.get("evaluation_start")),
        evaluation_end=_optional_str(payload.get("evaluation_end")),
        output_dir=_optional_str(payload.get("output_dir")),
        artifact_root=artifact_root,
        artifact_count=len(artifact_names),
        artifact_names=tuple(artifact_names),
        highlights_count=len(highlights or {}),
        warning_count=len(warnings or []),
    )


def dashboard_run_summary_rows(summary: DashboardRunSummary) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields(summary):
        value = getattr(summary, field.name)
        if value in (None, "", (), []):
            continue
        rows.append({"field": field.name, "value": _format_summary_value(value)})
    return rows


def _request_payload(request: Any) -> dict[str, Any]:
    if request is None:
        return {}
    if is_dataclass(request):
        return asdict(request)
    if isinstance(request, dict):
        return dict(request)
    return dict(getattr(request, "__dict__", {}))


def _resolve_primary_subject(*, service: str, strategies: list[str], resolved: dict[str, Any]) -> str | None:
    if service == "Strategy testbed":
        label = _optional_str(resolved.get("strategy_label"))
        if label:
            return label
    if len(strategies) == 1:
        return strategies[0]
    if strategies:
        return f"{len(strategies)} strategies"
    return None


def _resolve_strategies(payload: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    candidates = resolved.get("strategies", payload.get("strategies"))
    if isinstance(candidates, (list, tuple)):
        return [str(value) for value in candidates if str(value)]
    strategy = resolved.get("strategy", payload.get("strategy"))
    if strategy not in (None, ""):
        return [str(strategy)]
    return []


def _resolve_cleaning_method(payload: dict[str, Any], resolved: dict[str, Any]) -> str | None:
    return _optional_str(
        resolved.get("cleaning_method", resolved.get("method", payload.get("cleaning_method", payload.get("method"))))
    )


def _resolve_covariance_window(payload: dict[str, Any], resolved: dict[str, Any]) -> int | None:
    for key in ("covariance_window", "window"):
        if key in resolved:
            return _optional_int(resolved.get(key))
        if key in payload:
            return _optional_int(payload.get(key))
    windows = resolved.get("windows", payload.get("windows"))
    if isinstance(windows, (list, tuple)) and len(windows) == 1:
        return _optional_int(windows[0])
    return None


def _resolve_rebalance_frequency(payload: dict[str, Any], resolved: dict[str, Any]) -> str | None:
    return _optional_str(
        resolved.get(
            "rebalance_frequency",
            payload.get("rebalance_frequency"),
        )
    )


def _resolve_artifacts(artifacts: Any) -> tuple[str | None, list[str]]:
    if artifacts is None:
        return None, []
    root_dir = getattr(artifacts, "root_dir", None)
    files = getattr(artifacts, "files", None)
    if isinstance(artifacts, dict):
        if files is None:
            files = artifacts.get("files", artifacts)
        if root_dir is None:
            root_dir = artifacts.get("root_dir")
    artifact_names = sorted(str(name) for name in (files or {}))
    return _optional_str(root_dir), artifact_names


def _format_summary_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return ", ".join(str(item) for item in value)
    return value


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
