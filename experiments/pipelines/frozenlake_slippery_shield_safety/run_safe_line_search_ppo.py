"""Run the SafeLineSearch PPO FrozenLake slippery shield-safety baseline."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.frozenlake_slippery_shield_safety.core.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--mode", "downstream_safe_line_search"]))
