"""CLI wrapper for multi-seed launcher."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.orchestration.launch_multi_seed import main

if __name__ == "__main__":
    raise SystemExit(main())
