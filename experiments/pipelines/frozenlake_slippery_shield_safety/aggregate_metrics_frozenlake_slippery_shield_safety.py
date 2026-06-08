"""Aggregate FrozenLake slippery shield-safety metrics across seeds and export CSV + LaTeX."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.frozenlake_slippery_shield_safety.aggregate_metrics_frozenlake_safety import *  # noqa: F403
from experiments.pipelines.frozenlake_slippery_shield_safety.aggregate_metrics_frozenlake_safety import main


if __name__ == "__main__":
    raise SystemExit(main())
