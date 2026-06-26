"""Shared helpers duplicated across per-environment downstream adaptation methods."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml


def neutralize_task_feature(
    model: torch.nn.Sequential,
    task_feature_index: int,
    target_task_value: float,
) -> None:
    """Neutralize task-id contribution in the first linear layer for target task value.

    For first-layer pre-activation: z = Wx + b, with x_task = target_task_value,
    this applies:
        b <- b - W_task * target_task_value
        W_task <- 0
    so the task feature no longer shifts activations at adaptation start.
    """
    first = model[0]
    if not isinstance(first, torch.nn.Linear):
        raise ValueError("Expected first layer to be torch.nn.Linear for task-feature neutralization.")

    with torch.no_grad():
        w_task = first.weight[:, task_feature_index].clone()
        first.bias[:] = first.bias - w_task * target_task_value
        first.weight[:, task_feature_index] = 0.0


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
