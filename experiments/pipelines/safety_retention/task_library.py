"""Shared helpers for locating and reading the per-family task libraries.

Each environment family has one umbrella directory under safety_retention/ holding a
single ``settings/tasks/tasks.yaml``. A task block describes the ENVIRONMENT instance:
its concrete env id, constructor ``env_kwargs`` and a ``stochasticity`` block of dynamics
kwargs. ``synthesise_shield.py`` reads that block so the env it builds (and the shield it
computes) respect whatever stochasticity is written in the yaml; ``plot_shield.py`` then
inherits the same kwargs via the saved shield payload.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

FROZEN_LAKE_ENV = "FrozenLake-v1"
_SAFETY_RETENTION_ROOT = Path(__file__).resolve().parent


def environment_subdir(env_id: str) -> str:
    """Directory under safety_retention/ where this environment's outputs live.

    Environments are grouped by family under one umbrella directory: the version
    suffix is stripped so e.g. CustomColourBombGridWorld-v0/V2-v0/V3-v0 all resolve
    to 'CustomColourBombGridWorld', and 'FrozenLake-v1' resolves to 'FrozenLake'.
    """
    if env_id == FROZEN_LAKE_ENV:
        return "FrozenLake"
    base = re.sub(r"-v\d+$", "", env_id)  # drop the gym version suffix, e.g. '-v0'
    base = re.sub(r"V\d+$", "", base)  # drop the family-version marker, e.g. 'V2'
    return base


def tasks_library_path(env_id: str) -> Path:
    """Path to the shared tasks.yaml for ``env_id``'s family."""
    return _SAFETY_RETENTION_ROOT / environment_subdir(env_id) / "settings" / "tasks" / "tasks.yaml"


def load_masa_task(env_id: str, task_label: str) -> dict[str, Any]:
    """Return the task block ``task_label`` from ``env_id``'s family tasks.yaml.

    Raises a clear error if the library or the task is missing, or if the task's ``env``
    field disagrees with ``env_id`` (a sign of a mismatched --env/--task pair).
    """
    path = tasks_library_path(env_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No task library for {env_id} at {path}. Expected a shared tasks.yaml under "
            f"the '{environment_subdir(env_id)}' family directory.",
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if task_label not in data:
        raise KeyError(
            f"Task '{task_label}' not found in {path}. Available tasks: {sorted(data)}.",
        )
    block = dict(data[task_label] or {})
    block_env = block.get("env")
    if block_env is not None and str(block_env) != env_id:
        raise ValueError(
            f"Task '{task_label}' in {path} is defined for env '{block_env}', not '{env_id}'. "
            "Pass --env matching the task's env field.",
        )
    return block


def masa_env_kwargs(
    task_block: dict[str, Any],
    cli_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge a task block's constructor kwargs into a single dict for ``make_custom_masa_env``.

    Precedence (later wins): ``env_kwargs`` (non-dynamics args, e.g. layout) < ``stochasticity``
    (the dynamics block the user edits) < ``cli_override`` (an optional one-off --env-kwargs).
    Each source may be null/absent.
    """
    kwargs: dict[str, Any] = {}
    kwargs.update(task_block.get("env_kwargs") or {})
    kwargs.update(task_block.get("stochasticity") or {})
    kwargs.update(cli_override or {})
    return kwargs
