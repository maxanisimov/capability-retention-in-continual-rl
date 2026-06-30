"""Centralised random-seed control for reproducible runs."""

from __future__ import annotations

import random

import numpy as np

# Offsets applied to a run's base seed so the training env, evaluation env, and
# per-episode resets use distinct but deterministic seed streams. These replace
# the ``seed + 10000`` / ``+ 20000`` / ``+ 30000`` magic literals that were
# duplicated across the stage scripts.
TRAIN_SEED_OFFSET = 30_000
EVAL_SEED_OFFSET = 20_000
EPISODE_SEED_OFFSET = 10_000


def set_global_seeds(seed: int) -> None:
    """Seed ``random``, ``numpy`` and (if importable) ``torch``.

    Torch is imported lazily so this helper stays usable in contexts where torch
    is not required.
    """

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - torch is a hard dep in practice
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
