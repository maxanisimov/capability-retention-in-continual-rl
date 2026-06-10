"""Backward-compatible entrypoint for FrozenLake CSV-to-LaTeX conversion."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.behaviour_retention.frozenlake.core.analysis.csv_to_latex_table import main

if __name__ == "__main__":
    main()

