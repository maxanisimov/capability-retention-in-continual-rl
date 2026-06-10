"""Backward-compatible entrypoint for FrozenLake PPO sweeps."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.behaviour_retention.frozenlake.core.methods.sweep_scaled_ppo import CoordObsWrapper
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.sweep_scaled_ppo import DenseShapingWrapper
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.sweep_scaled_ppo import SafetyFlagWrapper
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.sweep_scaled_ppo import main
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.sweep_scaled_ppo import make_diagonal_source_map
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.sweep_scaled_ppo import make_env
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.sweep_scaled_ppo import schedule_for_size
from experiments.pipelines.behaviour_retention.frozenlake.core.methods.sweep_scaled_ppo import train_and_eval

__all__ = [
    "CoordObsWrapper",
    "DenseShapingWrapper",
    "SafetyFlagWrapper",
    "main",
    "make_diagonal_source_map",
    "make_env",
    "schedule_for_size",
    "train_and_eval",
]

if __name__ == "__main__":
    main()

