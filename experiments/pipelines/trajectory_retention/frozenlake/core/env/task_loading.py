"""Task/layout loading helpers for FrozenLake experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data)}.")
    return data


def _role_task_num(task_role: str) -> float:
    if task_role == "source":
        return 0.0
    if task_role == "downstream":
        return 1.0
    raise ValueError(f"task_role must be 'source' or 'downstream', got '{task_role}'.")


def _default_task_definitions_file(task_settings_file: Path) -> Path:
    if task_settings_file.name == "task_pipelines.yaml":
        return task_settings_file.with_name("task_definitions.yaml")
    return task_settings_file.parent / "task_definitions.yaml"


def _is_pipeline_entry(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("source"), dict)
        and isinstance(entry.get("downstream"), dict)
        and "env" in entry["source"]
        and "env" in entry["downstream"]
    )


def _normalize_env_cfg(
    cfg: dict[str, Any],
    *,
    task_role: str,
    append_task_id: bool,
    settings_format: str,
    task_pipelines_file: Path | None,
    task_definitions_file: Path | None,
    resolved_pipeline_name: str | None,
    resolved_definition_name: str,
) -> dict[str, Any]:
    env_map = cfg.get("env_map")
    if env_map is None:
        env_map = cfg.get("env1_map") if task_role == "source" else cfg.get("env2_map")
    if env_map is None:
        raise ValueError(f"Task definition '{resolved_definition_name}' does not define a FrozenLake map.")
    if not isinstance(env_map, list):
        raise ValueError(f"Task definition '{resolved_definition_name}' map must be a list of strings.")

    return {
        "env_id": cfg.get("env_id"),
        "env_kwargs": dict(cfg.get("env_kwargs", {}) or {}),
        "layout": cfg.get("layout"),
        "grid_size": cfg.get("grid_size"),
        "is_slippery": bool(cfg.get("is_slippery", False)),
        "max_episode_steps": int(cfg["max_episode_steps"]),
        "env_map": [str(row) for row in env_map],
        "task_num": _role_task_num(task_role),
        "append_task_id": bool(append_task_id),
        "_task_settings_format": settings_format,
        "_task_pipelines_file": (str(task_pipelines_file) if task_pipelines_file is not None else None),
        "_task_definitions_file": (str(task_definitions_file) if task_definitions_file is not None else None),
        "_resolved_pipeline_name": resolved_pipeline_name,
        "_resolved_definition_name": resolved_definition_name,
    }


def load_task_settings(settings_file: Path, setting_name: str, task_role: str) -> dict[str, Any]:
    """Load one source/downstream FrozenLake task from split or legacy settings."""
    if task_role not in {"source", "downstream"}:
        raise ValueError(f"task_role must be 'source' or 'downstream', got '{task_role}'.")
    if not settings_file.exists():
        raise FileNotFoundError(f"Task settings file not found: {settings_file}")

    all_settings = _load_yaml_mapping(settings_file)
    if any(_is_pipeline_entry(value) for value in all_settings.values()):
        if setting_name not in all_settings:
            if "default" in all_settings:
                setting_name = "default"
            else:
                raise ValueError(f"Task setting '{setting_name}' not found in {settings_file}.")
        pipeline_cfg = all_settings[setting_name]
        if not _is_pipeline_entry(pipeline_cfg):
            raise ValueError(f"Expected pipeline entry for '{setting_name}' in {settings_file}.")
        definition_name = pipeline_cfg[task_role]["env"]
        if not isinstance(definition_name, str) or not definition_name:
            raise ValueError(f"Expected non-empty string at {setting_name}:{task_role}.env")
        definitions_file = _default_task_definitions_file(settings_file)
        definitions = _load_yaml_mapping(definitions_file)
        if definition_name not in definitions:
            raise ValueError(f"Task definition '{definition_name}' not found in {definitions_file}.")
        definition_cfg = definitions[definition_name]
        if not isinstance(definition_cfg, dict):
            raise ValueError(f"Expected mapping for definition '{definition_name}' in {definitions_file}.")
        return _normalize_env_cfg(
            definition_cfg,
            task_role=task_role,
            append_task_id=bool(pipeline_cfg.get("append_task_id", True)),
            settings_format="split_pipeline",
            task_pipelines_file=settings_file,
            task_definitions_file=definitions_file,
            resolved_pipeline_name=setting_name,
            resolved_definition_name=definition_name,
        )

    if setting_name not in all_settings:
        raise ValueError(f"Task setting '{setting_name}' not found in {settings_file}.")
    cfg = all_settings[setting_name]
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping for task setting '{setting_name}' in {settings_file}.")
    return _normalize_env_cfg(
        cfg,
        task_role=task_role,
        append_task_id=True,
        settings_format="legacy_direct",
        task_pipelines_file=None,
        task_definitions_file=settings_file,
        resolved_pipeline_name=None,
        resolved_definition_name=setting_name,
    )


_load_task_settings = load_task_settings
