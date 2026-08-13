"""Axis-aligned boxes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Box:
    """An axis-aligned box represented by lower and upper tensors."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        lower = torch.as_tensor(self.lower, dtype=torch.float32)
        upper = torch.as_tensor(self.upper, dtype=torch.float32)
        if lower.shape != upper.shape:
            raise ValueError("Box lower and upper bounds must have the same shape.")
        if bool((lower > upper).any().item()):
            raise ValueError("Box lower bound exceeds upper bound.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dim(self) -> int:
        return int(self.lower.numel())

    @property
    def width(self) -> torch.Tensor:
        return self.upper - self.lower

    @property
    def max_width(self) -> float:
        return float(self.width.max().item())

    @property
    def center(self) -> torch.Tensor:
        return (self.lower + self.upper) / 2.0

    def split(self, dim: int | None = None) -> tuple["Box", "Box"]:
        """Bisect the box along ``dim`` or its widest dimension."""

        split_dim = int(torch.argmax(self.width).item()) if dim is None else int(dim)
        midpoint = self.center[split_dim]
        left_upper = self.upper.clone()
        right_lower = self.lower.clone()
        left_upper[split_dim] = midpoint
        right_lower[split_dim] = midpoint
        return Box(self.lower.clone(), left_upper), Box(right_lower, self.upper.clone())

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Uniformly sample points from the box."""

        shape = (int(n), self.dim)
        unit = torch.rand(shape, generator=generator, dtype=self.lower.dtype, device=self.lower.device)
        return self.lower + unit * self.width

    def corners(self) -> torch.Tensor:
        """Return all box corners."""

        points = []
        for mask in range(1 << self.dim):
            coords = [self.upper[i] if mask & (1 << i) else self.lower[i] for i in range(self.dim)]
            points.append(torch.stack(coords))
        return torch.stack(points, dim=0)

    def as_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lower.unsqueeze(0), self.upper.unsqueeze(0)

    @classmethod
    def from_lists(cls, bounds: list[list[float]]) -> "Box":
        """Build from ``[[lo0, hi0], [lo1, hi1], ...]``."""

        lower = torch.tensor([pair[0] for pair in bounds], dtype=torch.float32)
        upper = torch.tensor([pair[1] for pair in bounds], dtype=torch.float32)
        return cls(lower, upper)
