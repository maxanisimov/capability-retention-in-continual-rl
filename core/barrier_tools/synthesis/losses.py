"""Sampling losses for polynomial barrier training."""

from __future__ import annotations

import torch

from barrier_tools.barriers.base import BarrierFunction
from barrier_tools.dynamics.base import DiscreteTimeDynamics
from barrier_tools.policies.base import VerifiedPolicy


def initial_loss(barrier: BarrierFunction, states: torch.Tensor, eps_init: float) -> torch.Tensor:
    return torch.relu(float(eps_init) - barrier.value(states)).mean()


def unsafe_loss(barrier: BarrierFunction, states: torch.Tensor, eps_unsafe: float) -> torch.Tensor:
    return torch.relu(float(eps_unsafe) + barrier.value(states)).mean()


def invariance_loss(
    barrier: BarrierFunction,
    policy: VerifiedPolicy,
    dynamics: DiscreteTimeDynamics,
    states: torch.Tensor,
    *,
    alpha: float,
    eps_inv: float,
) -> torch.Tensor:
    h = barrier.value(states)
    actions = policy.forward(states)
    next_states = dynamics.step(states, actions)
    h_next = barrier.value(next_states)
    active = (h >= 0.0).float()
    violation = torch.relu((1.0 - float(alpha)) * h - h_next + float(eps_inv))
    if float(active.sum().item()) == 0.0:
        return violation.mean() * 0.0
    return (active * violation).sum() / active.sum()


def nonempty_regularizer(barrier: BarrierFunction, states: torch.Tensor) -> torch.Tensor:
    """Discourage barriers that make all sampled initial states negative."""

    return torch.relu(1e-3 - barrier.value(states).mean())
