"""CLI wrapper for Mountain Car source training."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.safe_crl.pipelines.envs.mountaincar.core.methods.source_train import main


if __name__ == "__main__":
    main()

