"""Deprecated compatibility entrypoint for :mod:`compare_pspo_initial_final_rewards`."""

from __future__ import annotations

import warnings

warnings.warn(
    "compare_pspo_adaptive_initial_final_rewards is deprecated; "
    "use compare_pspo_initial_final_rewards.",
    FutureWarning,
    stacklevel=2,
)

from projects.safe_policy_optimisation.scripts.compare_pspo_initial_final_rewards import *  # noqa: E402,F401,F403
from projects.safe_policy_optimisation.scripts.compare_pspo_initial_final_rewards import (
    main as _main,  # noqa: E402
)

if __name__ == "__main__":
    raise SystemExit(_main())
