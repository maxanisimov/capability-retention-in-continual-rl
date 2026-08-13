"""Policy interfaces."""

from __future__ import annotations

from typing import Protocol

import torch


class VerifiedPolicy(Protocol):
    """A deterministic policy with point and interval evaluation."""

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return a point action for each input state."""

    def interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower/upper action bounds for each state interval."""
