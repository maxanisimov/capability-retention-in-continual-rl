"""Backward-compatible entrypoint for relative aggregate metrics export."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.trajectory_retention.lunarlander.core.analysis.aggregate_layout_relative_metrics import main

if __name__ == "__main__":
    main()
