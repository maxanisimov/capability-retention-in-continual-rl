"""Backward-compatible entrypoint for LunarLander source training."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.env.env_factory import _make_lunarlander_env
from experiments.pipelines.lunarlander.core.env.task_loading import _load_task_settings
from experiments.pipelines.lunarlander.core.env.task_loading import _resolve_lunarlander_dynamics
from experiments.pipelines.lunarlander.core.methods.source_train import _plot_trajectory_grid
from experiments.pipelines.lunarlander.core.methods.source_train import build_actor_critic
from experiments.pipelines.lunarlander.core.methods.source_train import main

__all__ = [
    "_load_task_settings",
    "_resolve_lunarlander_dynamics",
    "_make_lunarlander_env",
    "build_actor_critic",
    "_plot_trajectory_grid",
    "main",
]

if __name__ == "__main__":
    main()
