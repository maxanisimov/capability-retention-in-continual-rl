"""CLI wrapper for FrozenLake post-hoc evaluation."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.frozenlake.core.eval.evaluate_post_hoc import main

if __name__ == "__main__":
    main()

