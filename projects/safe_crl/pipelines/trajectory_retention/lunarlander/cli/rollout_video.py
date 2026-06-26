"""CLI wrapper for policy rollout video export."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.safe_crl.pipelines.trajectory_retention.lunarlander.core.eval.rollout_video import main

if __name__ == "__main__":
    main()
