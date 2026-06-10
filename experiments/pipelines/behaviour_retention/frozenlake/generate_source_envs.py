"""Backward-compatible entrypoint for FrozenLake source environment generation."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.behaviour_retention.frozenlake.core.env.generate_source_envs import build_env_payload
from experiments.pipelines.behaviour_retention.frozenlake.core.env.generate_source_envs import main
from experiments.pipelines.behaviour_retention.frozenlake.core.env.generate_source_envs import make_diagonal_source_map

__all__ = ["build_env_payload", "main", "make_diagonal_source_map"]

if __name__ == "__main__":
    main()

