"""Backwards-compatible alias for :mod:`...utils.safe_rl`.

The helpers here are environment-agnostic, so the module was renamed to
``safe_rl``. This shim re-exports the public surface so existing imports of
``minipacman_safe_rl`` keep working; prefer importing from ``safe_rl`` (or the
focused ``io`` / ``shield`` / ``envs`` modules) in new code.
"""

from __future__ import annotations

from projects.safe_policy_optimisation.utils.safe_rl import *  # noqa: F401,F403
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: F401
    EpisodeMetrics,
    state_cost,
    write_json,
)
