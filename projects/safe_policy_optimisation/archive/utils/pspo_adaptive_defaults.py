"""Deprecated compatibility wrapper for :mod:`pspo_defaults`."""

from __future__ import annotations

import warnings

warnings.warn(
    "pspo_adaptive_defaults is deprecated; import pspo_defaults instead.",
    DeprecationWarning,
    stacklevel=2,
)

from projects.safe_policy_optimisation.utils.pspo_defaults import *  # noqa: E402,F401,F403
