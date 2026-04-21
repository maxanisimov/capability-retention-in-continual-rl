"""Backward-compatible multi-seed launcher for source training."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.pipelines.lunarlander.core.orchestration.launch_multi_seed import (
    main as launch_main,
)


def main() -> int:
    sys.argv = [sys.argv[0], "--mode", "source", *sys.argv[1:]]
    return launch_main()


if __name__ == "__main__":
    raise SystemExit(main())
