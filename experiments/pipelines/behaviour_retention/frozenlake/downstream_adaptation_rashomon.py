"""Backward-compatible entrypoint for Rashomon FrozenLake adaptation."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.behaviour_retention.frozenlake.core.methods.adapt_rashomon import compute_rashomon_bounds
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.adapt_rashomon import create_source_trajectory_rashomon_dataset
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.adapt_rashomon import main
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.adapt_rashomon import neutralize_task_feature

__all__ = [
    "compute_rashomon_bounds",
    "create_source_trajectory_rashomon_dataset",
    "main",
    "neutralize_task_feature",
]

if __name__ == "__main__":
    main()

