"""Compatibility delegate for :mod:`experiments.pipelines.safety.frozenlake_slippery.core.pipeline`."""

from __future__ import annotations

from importlib import import_module as _import_module
from pathlib import Path as _Path
import sys as _sys

for _parent in _Path(__file__).resolve().parents:
    if (_parent / "pyproject.toml").is_file() and (_parent / "experiments").is_dir():
        if str(_parent) not in _sys.path:
            _sys.path.insert(0, str(_parent))
        break

_CANONICAL_MODULE = "experiments.pipelines.safety.frozenlake_slippery.core.pipeline"
_module = _import_module(_CANONICAL_MODULE)

if __name__ == "__main__":
    _main = getattr(_module, "main", None)
    if _main is None:
        raise SystemExit(f"{_CANONICAL_MODULE} does not define main().")
    raise SystemExit(_main())

_sys.modules[__name__] = _module
globals().update(_module.__dict__)
