"""Backward-compatible entrypoint for task video generation."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.eval.generate_task_video import main

if __name__ == "__main__":
    main()

