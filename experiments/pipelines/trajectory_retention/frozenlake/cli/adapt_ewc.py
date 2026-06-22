"""CLI wrapper for EWC FrozenLake adaptation."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.trajectory_retention.frozenlake.core.methods.adapt_ewc import main

if __name__ == "__main__":
    main()

