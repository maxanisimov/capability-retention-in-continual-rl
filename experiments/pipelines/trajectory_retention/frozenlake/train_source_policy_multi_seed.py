"""Backward-compatible multi-seed source launcher for FrozenLake."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.trajectory_retention.frozenlake.core.orchestration.launch_multi_seed import main_with_mode

if __name__ == "__main__":
    raise SystemExit(main_with_mode("source"))

