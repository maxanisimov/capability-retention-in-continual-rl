"""Settings for the FrozenLake safety pipeline."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


LAYOUT = "diagonal_4x4"

_SETTINGS_ROOT = Path(__file__).resolve().parents[1] / "settings"
SOURCE_SETTINGS_FILE = _SETTINGS_ROOT / "source" / "train_source_policy_settings.yaml"
ADAPT_PPO_SETTINGS_FILE = _SETTINGS_ROOT / "adaptation" / "ppo.yaml"
ADAPT_EWC_SETTINGS_FILE = _SETTINGS_ROOT / "adaptation" / "ewc.yaml"
ADAPT_RASHOMON_SETTINGS_FILE = _SETTINGS_ROOT / "adaptation" / "rashomon.yaml"
TASKS_LIBRARY_FILE = _SETTINGS_ROOT / "tasks" / "tasks.yaml"
PIPELINES_SETTINGS_FILE = _SETTINGS_ROOT / "tasks" / "pipelines.yaml"


def _load_layout_block(path: Path, layout: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or layout not in data:
        raise ValueError(f"Expected layout '{layout}' in {path}.")
    block = data[layout]
    if not isinstance(block, dict):
        raise ValueError(f"Expected dict settings for layout '{layout}' in {path}.")
    return block


def _normalise_task_block(block: dict[str, Any]) -> dict[str, Any]:
    """Flatten a tasks.yaml task block to (map, max_episode_steps, deterministic,
    slip_probability).

    Accepts the shared family schema (env/env_kwargs/stochasticity, in line with the
    other safety environments) and falls back to the legacy flat layout, so both forms
    load identically.
    """
    env_kwargs = block.get("env_kwargs") or {}
    stochasticity = block.get("stochasticity") or {}

    def pick(key: str, default: Any) -> Any:
        for source in (block, env_kwargs, stochasticity):
            if key in source:
                return source[key]
        return default

    return {
        "map": pick("map", None),
        "max_episode_steps": pick("max_episode_steps", None),
        "deterministic": pick("deterministic", True),
        "slip_probability": float(pick("slip_probability", 0.0)),
    }


def _load_pipeline_tasks(layout: str) -> dict[str, Any]:
    pipeline_block = _load_layout_block(PIPELINES_SETTINGS_FILE, layout)
    source_key = pipeline_block["source"]["task"]
    downstream_key = pipeline_block["downstream"]["task"]
    source_task = _normalise_task_block(_load_layout_block(TASKS_LIBRARY_FILE, source_key))
    downstream_task = _normalise_task_block(_load_layout_block(TASKS_LIBRARY_FILE, downstream_key))

    source_steps = source_task["max_episode_steps"]
    downstream_steps = downstream_task["max_episode_steps"]
    if source_steps != downstream_steps:
        raise ValueError(
            f"Pipeline '{layout}' has mismatched max_episode_steps between tasks "
            f"'{source_key}' ({source_steps}) and '{downstream_key}' ({downstream_steps})."
        )

    source_deterministic = source_task.get("deterministic", True)
    downstream_deterministic = downstream_task.get("deterministic", True)
    if source_deterministic != downstream_deterministic:
        raise ValueError(
            f"Pipeline '{layout}' has mismatched deterministic between tasks "
            f"'{source_key}' ({source_deterministic}) and '{downstream_key}' ({downstream_deterministic})."
        )

    source_slip = float(source_task.get("slip_probability", 0.0))
    downstream_slip = float(downstream_task.get("slip_probability", 0.0))
    if source_slip != downstream_slip:
        raise ValueError(
            f"Pipeline '{layout}' has mismatched slip_probability between tasks "
            f"'{source_key}' ({source_slip}) and '{downstream_key}' ({downstream_slip})."
        )
    if not 0.0 <= source_slip <= 1.0:
        raise ValueError(
            f"Pipeline '{layout}' task '{source_key}' has slip_probability={source_slip} outside [0, 1]."
        )
    if source_deterministic and source_slip != 0.0:
        raise ValueError(
            f"Pipeline '{layout}' task '{source_key}' is deterministic but has "
            f"slip_probability={source_slip}; deterministic tasks must have slip_probability=0."
        )

    return {
        "source_map": source_task["map"],
        "downstream_map": downstream_task["map"],
        "max_episode_steps": source_steps,
        "deterministic": source_deterministic,
        "slip_probability": source_slip,
    }


@lru_cache(maxsize=None)
def _cached_settings_for_layout(layout: str) -> dict[str, Any]:
    return {
        "layout": layout,
        "source": _load_layout_block(SOURCE_SETTINGS_FILE, layout),
        "adaptation_ppo": _load_layout_block(ADAPT_PPO_SETTINGS_FILE, layout),
        "adaptation_ewc": _load_layout_block(ADAPT_EWC_SETTINGS_FILE, layout),
        "adaptation_rashomon": _load_layout_block(ADAPT_RASHOMON_SETTINGS_FILE, layout),
        "tasks": _load_pipeline_tasks(layout),
        "settings_files": {
            "source": str(SOURCE_SETTINGS_FILE),
            "adaptation_ppo": str(ADAPT_PPO_SETTINGS_FILE),
            "adaptation_ewc": str(ADAPT_EWC_SETTINGS_FILE),
            "adaptation_rashomon": str(ADAPT_RASHOMON_SETTINGS_FILE),
            "tasks": str(TASKS_LIBRARY_FILE),
            "pipelines": str(PIPELINES_SETTINGS_FILE),
        },
    }


def settings_for_layout(layout: str) -> dict[str, Any]:
    """Return a defensive copy of the settings for the given layout."""
    return copy.deepcopy(_cached_settings_for_layout(layout))


def load_task_definition(task_name: str) -> dict[str, Any]:
    """Return a single task block (map, max_episode_steps, deterministic, slip_probability)
    from tasks.yaml by its task-library key (e.g. 'diagonal_4x4_stochastic_source')."""
    return _normalise_task_block(
        copy.deepcopy(_load_layout_block(TASKS_LIBRARY_FILE, task_name)),
    )


def frozenlake_safety_diagonal_4x4_settings() -> dict[str, Any]:
    """Return a defensive copy of the FrozenLake safety diagonal_4x4 settings."""
    return settings_for_layout("diagonal_4x4")
