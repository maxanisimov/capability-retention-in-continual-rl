"""Generate FrozenLake shield-safety source/downstream initial-frame figures."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.frozenlake_shield_safety.core.analysis.plot_initial_frames import (
    main,
)


if __name__ == "__main__":
    main()
