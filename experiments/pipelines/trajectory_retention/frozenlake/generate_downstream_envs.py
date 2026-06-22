"""Backward-compatible entrypoint for FrozenLake downstream environment generation."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.trajectory_retention.frozenlake.core.env.generate_downstream_envs import create_downstream_map
from experiments.pipelines.trajectory_retention.frozenlake.core.env.generate_downstream_envs import main
from experiments.pipelines.trajectory_retention.frozenlake.core.env.generate_downstream_envs import map_is_solvable
from experiments.pipelines.trajectory_retention.frozenlake.core.env.generate_downstream_envs import no_hole_next_to_start_or_goal
from experiments.pipelines.trajectory_retention.frozenlake.core.env.generate_downstream_envs import protect_start_and_goal_neighbors
from experiments.pipelines.trajectory_retention.frozenlake.core.env.generate_downstream_envs import swap_frozenlake_cells

__all__ = [
    "create_downstream_map",
    "main",
    "map_is_solvable",
    "no_hole_next_to_start_or_goal",
    "protect_start_and_goal_neighbors",
    "swap_frozenlake_cells",
]

if __name__ == "__main__":
    main()

