"""Backward-compatible module re-export for tunable LunarLander env."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.env.tunable_lunarlander import *  # noqa: F401,F403
