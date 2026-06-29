"""Configuration helpers for safe-policy-optimisation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_ROOT = PROJECT_ROOT / "settings"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
TASKS_FILE = SETTINGS_ROOT
PIPELINES_FILE = SETTINGS_ROOT


PIPELINE_SETTING_SECTIONS = (
    "output",
    "runtime",
    "shield_synthesis",
    "safety",
    "safe_rl_baselines",
    "training",
    "rashomon_set",
    "early_stopping",
    "evaluation",
    "monitoring",
    "stages",
)


def cli_supplied_flags(argv: list[str]) -> set[str]:
    """Return argparse-style flag names explicitly supplied on the command line."""

    flags: set[str] = set()
    for token in argv:
        if token.startswith("--"):
            flags.add(token.split("=", 1)[0][2:].replace("-", "_"))
    return flags


def load_yaml_settings(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Settings YAML must contain a mapping, got {type(payload).__name__}: {path}")
    return payload


def _load_named_registry(path: Path, root_key: str) -> dict[str, Any]:
    payload = load_yaml_settings(path)
    registry = payload.get(root_key)
    if not isinstance(registry, dict):
        raise ValueError(f"{path} must contain a top-level {root_key!r} mapping.")
    return registry


def _registry_files(path: Path, filename: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Settings path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Settings path must be a file or directory: {path}")
    direct = path / filename
    if direct.exists():
        return [direct]
    return sorted(child for child in path.glob(f"*/{filename}") if child.is_file())


def _load_merged_named_registry(path: Path, root_key: str, filename: str) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    files = _registry_files(path, filename)
    if not files:
        raise FileNotFoundError(f"No {filename} files found under {path}")
    for file_path in files:
        for name, value in _load_named_registry(file_path, root_key).items():
            if name in registry:
                raise ValueError(f"Duplicate {root_key[:-1]} {name!r} found while reading {file_path}")
            registry[name] = value
    return registry


def registry_source_file(path: Path, *, filename: str, root_key: str, name: str) -> Path:
    """Return the YAML file containing a named registry entry."""

    for file_path in _registry_files(path, filename):
        registry = _load_named_registry(file_path, root_key)
        if name in registry:
            return file_path
    raise KeyError(f"Could not find {root_key[:-1]} {name!r} under {path}")


def load_task_registry(path: Path = TASKS_FILE) -> dict[str, Any]:
    """Load task definitions from one tasks.yaml file or all environment task files."""

    return _load_merged_named_registry(path, "tasks", "tasks.yaml")


def load_pipeline_registry(path: Path = PIPELINES_FILE) -> dict[str, Any]:
    """Load pipeline definitions from one pipelines.yaml file or all environment pipeline files."""

    return _load_merged_named_registry(path, "pipelines", "pipelines.yaml")


def _merge_section(target: dict[str, Any], section_name: str, section: Any, *, source: Path) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        raise ValueError(f"{source}: pipeline section {section_name!r} must be a mapping.")
    target.update(section)


def flatten_pipeline_definition(pipeline: dict[str, Any], *, source: Path = PIPELINES_FILE) -> dict[str, Any]:
    """Flatten structured pipeline settings into parser-compatible keys."""

    flat: dict[str, Any] = {}
    for section_name in PIPELINE_SETTING_SECTIONS:
        _merge_section(flat, section_name, pipeline.get(section_name), source=source)
    return flat


def compose_pipeline_settings(
    pipeline_name: str,
    *,
    task_name: str | None = None,
    tasks_file: Path = TASKS_FILE,
    pipelines_file: Path = PIPELINES_FILE,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compose flat argparse settings from a task and a structured pipeline config."""

    pipelines = load_pipeline_registry(pipelines_file)
    if pipeline_name not in pipelines:
        raise KeyError(f"Unknown pipeline {pipeline_name!r}. Available: {', '.join(sorted(pipelines))}")
    pipeline = pipelines[pipeline_name]
    if not isinstance(pipeline, dict):
        raise ValueError(f"{pipelines_file}: pipeline {pipeline_name!r} must be a mapping.")

    selected_task = task_name or pipeline.get("default_task")
    if not selected_task:
        raise ValueError(f"{pipelines_file}: pipeline {pipeline_name!r} must define default_task or receive a task.")

    tasks = load_task_registry(tasks_file)
    if selected_task not in tasks:
        raise KeyError(f"Unknown task {selected_task!r}. Available: {', '.join(sorted(tasks))}")
    task = tasks[selected_task]
    if not isinstance(task, dict):
        raise ValueError(f"{tasks_file}: task {selected_task!r} must be a mapping.")

    env_kwargs = task.get("env_kwargs") or {}
    if not isinstance(env_kwargs, dict):
        raise ValueError(f"{tasks_file}: task {selected_task!r} env_kwargs must be a mapping or null.")

    flat = {
        "task": selected_task,
        "env_id": task.get("env_id"),
        "env_kwargs": env_kwargs,
        "max_episode_steps": task.get("max_episode_steps"),
    }
    if "ghost_rand_prob" in env_kwargs:
        flat["ghost_rand_prob"] = env_kwargs["ghost_rand_prob"]
    flat.update(flatten_pipeline_definition(pipeline, source=pipelines_file))
    return flat, pipeline, task


def coerce_setting_value(key: str, current: Any, value: Any) -> Any:
    """Coerce YAML values to the type expected by an argparse namespace."""

    path_settings = {
        "output_dir",
        "settings_file",
        "shield_path",
        "rashomon_dir",
        "checkpoint",
        "run_dir",
        "tensorboard_log_dir",
    }
    if isinstance(current, Path) or key in path_settings:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path
    return value


def apply_settings_to_namespace(
    args: argparse.Namespace,
    settings: dict[str, Any],
    *,
    settings_file: Path,
    explicit_flags: set[str] | None = None,
    ignored_keys: set[str] | None = None,
) -> argparse.Namespace:
    """Apply YAML settings to a parsed argparse namespace.

    Explicit CLI flags win over YAML. Unknown keys fail fast so misspelled
    settings do not silently produce a different experiment.
    """

    explicit = set(explicit_flags or set())
    ignored = {"description", *(ignored_keys or set())}
    for key, value in settings.items():
        if key in ignored:
            continue
        if not hasattr(args, key):
            raise ValueError(f"Unknown setting {key!r} in {settings_file}")
        if key in explicit:
            continue
        setattr(args, key, coerce_setting_value(key, getattr(args, key), value))
    args.training_settings_file = settings_file
    args.training_settings = settings
    return args


def load_and_apply_settings(
    args: argparse.Namespace,
    settings_file: Path,
    *,
    explicit_flags: set[str] | None = None,
    ignored_keys: set[str] | None = None,
) -> argparse.Namespace:
    """Load YAML settings from disk and apply them to an argparse namespace."""

    if not settings_file.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_file}")
    settings = load_yaml_settings(settings_file)
    return apply_settings_to_namespace(
        args,
        settings,
        settings_file=settings_file,
        explicit_flags=explicit_flags,
        ignored_keys=ignored_keys,
    )
