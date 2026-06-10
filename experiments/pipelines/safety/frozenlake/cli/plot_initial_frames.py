"""CLI wrapper for FrozenLake shield-safety initial-frame figure generation."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.safety.frozenlake.core.analysis.plot_initial_frames import (
    main,
)


if __name__ == "__main__":
    main()
