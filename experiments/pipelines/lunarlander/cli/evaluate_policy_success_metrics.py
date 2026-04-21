"""CLI wrapper for success/failure metrics policy evaluation."""

import importlib
import importlib.util
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CORE_MODULE_NAME = "experiments.pipelines.lunarlander.core.eval.evaluate_policy_success_metrics"
_CORE_MODULE_PATH = (
    _REPO_ROOT
    / "experiments"
    / "pipelines"
    / "lunarlander"
    / "core"
    / "eval"
    / "evaluate_policy_success_metrics.py"
)


def _load_main():
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(_CORE_MODULE_NAME)
        main_fn = getattr(module, "main", None)
        if callable(main_fn):
            return main_fn
    except Exception:
        pass

    # Fallback: remove stale bytecode and load directly from source file.
    pycache_dir = _CORE_MODULE_PATH.parent / "__pycache__"
    if pycache_dir.exists():
        for pyc_file in pycache_dir.glob("evaluate_policy_success_metrics.cpython-*.pyc"):
            try:
                pyc_file.unlink()
            except OSError:
                pass

    spec = importlib.util.spec_from_file_location("_lunarlander_eval_success_metrics_cli_fallback", _CORE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for: {_CORE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        raise ImportError(
            "Unable to load callable `main` from "
            f"{_CORE_MODULE_NAME} ({_CORE_MODULE_PATH})."
        )
    return main_fn

if __name__ == "__main__":
    _load_main()()
