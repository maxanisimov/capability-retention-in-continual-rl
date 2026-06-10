"""Entrypoint for FrozenLake Slippery Shield Safety layout metric aggregation."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.safety.frozenlake_slippery.core.analysis.aggregate_layout_metrics import main

if __name__ == "__main__":
    main()
