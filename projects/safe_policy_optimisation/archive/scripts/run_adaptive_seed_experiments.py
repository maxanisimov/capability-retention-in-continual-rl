"""Deprecated compatibility entrypoint for :mod:`run_pspo_seed_experiments`."""

from __future__ import annotations

import warnings

warnings.warn(
    "run_adaptive_seed_experiments is deprecated; use run_pspo_seed_experiments.",
    FutureWarning,
    stacklevel=2,
)

from projects.safe_policy_optimisation.scripts.run_pspo_seed_experiments import *  # noqa: E402,F401,F403
from projects.safe_policy_optimisation.scripts.run_pspo_seed_experiments import (
    main as _main,  # noqa: E402
)

if __name__ == "__main__":
    raise SystemExit(_main())
