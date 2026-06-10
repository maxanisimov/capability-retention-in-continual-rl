"""Synthesize and plot a FrozenLake slippery shield for one pipeline layout."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.safety.frozenlake_slippery.core.analysis.plot_synthesised_shield import (
    main,
)


if __name__ == "__main__":
    main()
