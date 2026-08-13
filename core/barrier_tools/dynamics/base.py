"""Base interfaces for discrete-time dynamics."""

from __future__ import annotations

from typing import Protocol

import torch


class DiscreteTimeDynamics(Protocol):
    """A deterministic discrete-time system with interval propagation."""

    state_dim: int
    action_dim: int

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the next state for a point state/action batch."""

    def interval_step(
        self,
        state_lower: torch.Tensor,
        state_upper: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower/upper next-state bounds for an interval box."""
