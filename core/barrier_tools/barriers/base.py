"""Barrier-function interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch


class BarrierFunction(Protocol):
    """A scalar barrier using ``h(s) >= 0`` for the certified safe set."""

    def value(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the barrier at point states."""

    def interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower/upper bounds over a state interval."""

    def save(self, path: Path) -> None:
        """Save the barrier parameters."""
