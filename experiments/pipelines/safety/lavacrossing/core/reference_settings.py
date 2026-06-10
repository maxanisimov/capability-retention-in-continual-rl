"""Settings for the LavaCrossing shield-safety pipeline."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


LAYOUT = "corridor_7x7_deterministic"

_SETTINGS_ROOT = Path(__file__).resolve().parents[1] / "settings"
SOURCE_SETTINGS_FILE = _SETTINGS_ROOT / "source" / "train_source_policy_settings.yaml"
ADAPT_PPO_SETTINGS_FILE = _SETTINGS_ROOT / "adaptation" / "ppo.yaml"
ADAPT_EWC_SETTINGS_FILE = _SETTINGS_ROOT / "adaptation" / "ewc.yaml"
ADAPT_RASHOMON_SETTINGS_FILE = _SETTINGS_ROOT / "adaptation" / "rashomon.yaml"
TASKS_SETTINGS_FILE = _SETTINGS_ROOT / "tasks" / "envs.yaml"
SHIELD_SETTINGS_FILE = _SETTINGS_ROOT / "shield.yaml"


def _load_layout_block(path: Path, layout: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or layout not in data:
        raise ValueError(f"Expected layout '{layout}' in {path}.")
    block = data[layout]
    if not isinstance(block, dict):
        raise ValueError(f"Expected dict settings for layout '{layout}' in {path}.")
    return block


@lru_cache(maxsize=None)
def _cached_settings_for_layout(layout: str) -> dict[str, Any]:
    return {
        "layout": layout,
        "source": _load_layout_block(SOURCE_SETTINGS_FILE, layout),
        "adaptation_ppo": _load_layout_block(ADAPT_PPO_SETTINGS_FILE, layout),
        "adaptation_ewc": _load_layout_block(ADAPT_EWC_SETTINGS_FILE, layout),
        "adaptation_rashomon": _load_layout_block(ADAPT_RASHOMON_SETTINGS_FILE, layout),
        "tasks": _load_layout_block(TASKS_SETTINGS_FILE, layout),
        "shield": _load_layout_block(SHIELD_SETTINGS_FILE, layout),
        "settings_files": {
            "source": str(SOURCE_SETTINGS_FILE),
            "adaptation_ppo": str(ADAPT_PPO_SETTINGS_FILE),
            "adaptation_ewc": str(ADAPT_EWC_SETTINGS_FILE),
            "adaptation_rashomon": str(ADAPT_RASHOMON_SETTINGS_FILE),
            "tasks": str(TASKS_SETTINGS_FILE),
            "shield": str(SHIELD_SETTINGS_FILE),
        },
    }


def settings_for_layout(layout: str) -> dict[str, Any]:
    """Return a defensive copy of the settings for the given layout."""
    return copy.deepcopy(_cached_settings_for_layout(layout))


def lavacrossing_shield_safety_corridor_7x7_deterministic_settings() -> dict[str, Any]:
    """Return the default LavaCrossing shield safety settings."""
    return settings_for_layout("corridor_7x7_deterministic")
