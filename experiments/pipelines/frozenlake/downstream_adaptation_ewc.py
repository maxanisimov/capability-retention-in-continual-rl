"""Backward-compatible entrypoint for EWC FrozenLake adaptation."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.frozenlake.core.methods.adapt_ewc import main
from experiments.pipelines.frozenlake.core.methods.adapt_ewc import neutralize_task_feature

__all__ = ["main", "neutralize_task_feature"]

if __name__ == "__main__":
    main()

