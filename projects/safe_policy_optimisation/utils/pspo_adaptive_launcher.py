"""Deprecated compatibility wrapper for :mod:`pspo_launcher`."""

from __future__ import annotations

import warnings

warnings.warn(
    "pspo_adaptive_launcher is deprecated; import pspo_launcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

from projects.safe_policy_optimisation.utils.pspo_launcher import *  # noqa: E402,F401,F403
