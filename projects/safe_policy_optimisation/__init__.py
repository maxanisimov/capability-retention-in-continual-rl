"""Safe policy optimisation project package.

Importing this package configures a writable, headless matplotlib config
directory once, so individual stage modules no longer need to set
``MPLCONFIGDIR`` themselves. This runs before any submodule body (and therefore
before matplotlib is imported by stages), regardless of whether a stage is run
via ``python -m`` or ``python run_experiment.py``.
"""

from __future__ import annotations

import os
import tempfile

# Matplotlib writes a font cache / config on first import; on shared machines the
# default ``$HOME/.config/matplotlib`` may be unwritable. Point it at a temp dir
# unless the caller already configured one.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig")
)
