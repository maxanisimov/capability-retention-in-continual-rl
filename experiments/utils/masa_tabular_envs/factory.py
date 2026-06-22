"""Factory helpers for custom MASA-style tabular environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


def make_custom_masa_env(
    env_id: str,
    *,
    max_episode_steps: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a registered custom MASA tabular environment.

    Importing this module registers the local ``Custom*`` Gymnasium IDs via the
    package ``__init__``.
    """

    import experiments.utils.masa_tabular_envs  # noqa: F401

    kwargs = dict(env_kwargs or {})
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, max_episode_steps=max_episode_steps, **kwargs)
