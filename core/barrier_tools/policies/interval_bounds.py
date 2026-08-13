"""Interval propagation for small feed-forward PyTorch policies."""

from __future__ import annotations

import torch
from torch import nn

from barrier_tools.verification.interval import affine_interval


def bound_module(
    module: nn.Module,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate interval bounds through one supported module."""

    if isinstance(module, nn.Linear):
        return affine_interval(lower, upper, module.weight, module.bias)
    if isinstance(module, nn.ReLU):
        return torch.relu(lower), torch.relu(upper)
    if isinstance(module, nn.Tanh):
        return torch.tanh(lower), torch.tanh(upper)
    if isinstance(module, nn.Identity):
        return lower, upper
    raise TypeError(f"Unsupported policy layer for interval bounds: {type(module).__name__}")


def bound_sequential(
    model: nn.Sequential,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate interval bounds through a supported sequential model."""

    out_l, out_u = lower, upper
    for module in model:
        out_l, out_u = bound_module(module, out_l, out_u)
    return out_l, out_u
