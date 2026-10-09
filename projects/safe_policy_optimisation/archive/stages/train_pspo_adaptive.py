"""Deprecated compatibility wrapper for :mod:`train_pspo`."""

from __future__ import annotations

import warnings

warnings.warn(
    "train_pspo_adaptive is deprecated; import train_pspo instead.",
    DeprecationWarning,
    stacklevel=2,
)

from projects.safe_policy_optimisation.stages.train_pspo import *  # noqa: E402,F401,F403
from projects.safe_policy_optimisation.stages.train_pspo import main as _main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(_main())
