"""Backward-compatible entrypoint for FrozenLake source training."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.trajectory_retention.frozenlake.core.methods.source_train import build_actor_critic
from experiments.pipelines.trajectory_retention.frozenlake.core.methods.source_train import main
from experiments.pipelines.trajectory_retention.frozenlake.core.methods.source_train import make_env_from_layout

__all__ = ["build_actor_critic", "make_env_from_layout", "main"]

if __name__ == "__main__":
    main()

