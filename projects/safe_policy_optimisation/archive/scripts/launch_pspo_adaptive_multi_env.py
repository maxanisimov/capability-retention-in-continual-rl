"""Deprecated compatibility entrypoint for :mod:`launch_pspo_multi_env`."""

from __future__ import annotations

import warnings

warnings.warn(
    "launch_pspo_adaptive_multi_env is deprecated; use launch_pspo_multi_env.",
    FutureWarning,
    stacklevel=2,
)

from projects.safe_policy_optimisation.scripts.launch_pspo_multi_env import *  # noqa: E402,F401,F403
from projects.safe_policy_optimisation.scripts.launch_pspo_multi_env import (
    main as _main,  # noqa: E402
)

if __name__ == "__main__":
    raise SystemExit(_main())
