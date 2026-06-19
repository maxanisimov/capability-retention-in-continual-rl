"""Backward-compatible entrypoint for Mountain Car source training."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.mountaincar.core.env.env_factory import make_mountaincar_env
from experiments.pipelines.mountaincar.core.methods.source_train import (
    build_actor_critic,
    main,
)

__all__ = ["build_actor_critic", "main", "make_mountaincar_env"]


if __name__ == "__main__":
    main()

