"""Shared Rashomon-set command-line helpers."""

from __future__ import annotations

import argparse
from typing import Any, Literal

import numpy as np

RashomonBatchSize = int | Literal["auto"]


def parse_rashomon_batch_size(value: str) -> RashomonBatchSize:
    """Parse ``auto`` (or legacy ``all``) and positive integer batch sizes."""

    normalized = str(value).strip().lower()
    if normalized in {"auto", "all"}:
        return "auto"
    try:
        batch_size = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected 'auto' or a positive integer"
        ) from exc
    if batch_size <= 0:
        raise argparse.ArgumentTypeError("expected 'auto' or a positive integer")
    return batch_size


def safe_behaviour_dataset_size(shield_mask: Any) -> int:
    """Count shield states having at least one safe action."""

    mask = np.asarray(shield_mask)
    if mask.ndim != 2:
        raise ValueError(
            f"Expected shield mask shape (n_states, n_actions), got {mask.shape}."
        )
    dataset_size = int(np.count_nonzero(mask.astype(bool).any(axis=1)))
    if dataset_size == 0:
        raise ValueError("Shield contains no states with at least one safe action.")
    return dataset_size


def resolve_rashomon_batch_size(
    setting: RashomonBatchSize,
    shield_mask: Any,
) -> int:
    """Resolve ``auto`` to the complete safe-behaviour demonstration dataset."""

    if setting == "auto":
        return safe_behaviour_dataset_size(shield_mask)
    batch_size = int(setting)
    if batch_size <= 0:
        raise ValueError("Rashomon batch size must be positive.")
    return batch_size
