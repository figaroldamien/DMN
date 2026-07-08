from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class WorkspaceDefaults:
    config_path: str
    universe: str
    start: str
    evaluation_start: str | None
    evaluation_end: str | None


@dataclass(frozen=True)
class WorkspaceContext:
    config_path: str
    config_defaults: dict[str, Any]
    universe_group: str | None
    universe: str
    start: str
    evaluation_start: str | None
    evaluation_end: str | None
    refresh_pending: bool


def workspace_defaults_from_config(
    config_defaults: Mapping[str, Any],
    *,
    default_config_path: str,
    fallback_universe: str,
) -> WorkspaceDefaults:
    universe_defaults = config_defaults.get("universe", {})
    evaluation_defaults = config_defaults.get("evaluation", {})
    return WorkspaceDefaults(
        config_path=default_config_path,
        universe=str(universe_defaults.get("name", fallback_universe) or fallback_universe),
        start=str(universe_defaults.get("start", "") or ""),
        evaluation_start=_optional_str(evaluation_defaults.get("evaluation_start")),
        evaluation_end=_optional_str(evaluation_defaults.get("evaluation_end")),
    )


def normalize_workspace_selection(
    *,
    universe_groups: Mapping[str, list[str]],
    fallback_universe_options: list[str],
    universe_default: str,
    stored_group: str | None,
    stored_universe: str | None,
) -> tuple[str, list[str], str]:
    group_names = [name for name, options in universe_groups.items() if options]
    if not group_names:
        group_names = ["Universe"]
        universe_groups = {"Universe": fallback_universe_options}

    resolved_group = stored_group if stored_group in group_names else _default_universe_group(universe_default, universe_groups, group_names[0])
    group_options = universe_groups.get(resolved_group, fallback_universe_options) or fallback_universe_options
    resolved_universe = stored_universe if stored_universe in group_options else (
        universe_default if universe_default in group_options else group_options[0]
    )
    return resolved_group, group_options, resolved_universe


def workspace_mode_uses_shared_controls(mode: str) -> bool:
    return mode not in {"Guide", "Workspace"}


def build_workspace_context(
    *,
    config_path: str,
    config_defaults: Mapping[str, Any],
    universe_group: str | None,
    universe: str,
    start: str,
    evaluation_start: str | None,
    evaluation_end: str | None,
    refresh_pending: bool,
) -> WorkspaceContext:
    return WorkspaceContext(
        config_path=config_path,
        config_defaults=dict(config_defaults),
        universe_group=universe_group,
        universe=universe,
        start=start,
        evaluation_start=_optional_str(evaluation_start),
        evaluation_end=_optional_str(evaluation_end),
        refresh_pending=bool(refresh_pending),
    )


def _optional_str(value: Any) -> str | None:
    if value in (None, "", "None"):
        return None
    return str(value)


def _default_universe_group(
    universe_name: str,
    universe_groups: Mapping[str, list[str]],
    fallback_group: str,
) -> str:
    for group_name, options in universe_groups.items():
        if universe_name in options:
            return group_name
    return fallback_group
