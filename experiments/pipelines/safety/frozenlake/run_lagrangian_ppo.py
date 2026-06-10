"""Run the Lagrangian PPO FrozenLake shield-safety baseline."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.safety.frozenlake.core.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--mode", "downstream_lagrangian"]))
