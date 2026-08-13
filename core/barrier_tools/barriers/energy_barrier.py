"""Analytic energy-inspired MountainCar barrier candidate."""

from __future__ import annotations

from pathlib import Path

import torch

from barrier_tools.verification.interval import (
    add_interval,
    cos_interval,
    scalar_mul_interval,
)


class EnergyBarrier:
    """Energy-inspired candidate with boundary anchored at ``p_safe`` and ``v=0``."""

    def __init__(self, p_safe: float = -1.0) -> None:
        self.p_safe = float(p_safe)
        self.c = -self._raw(torch.tensor([[self.p_safe, 0.0]], dtype=torch.float32))[0]

    @staticmethod
    def _raw(state: torch.Tensor) -> torch.Tensor:
        p = state[..., 0]
        v = state[..., 1]
        return -0.5 * v.pow(2) - (0.0025 / 3.0) * torch.sin(3.0 * p) + 0.0015 * p

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.c.to(dtype=state.dtype, device=state.device) + self._raw(state)

    def interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_l = lower[..., 0]
        p_u = upper[..., 0]
        v_l = lower[..., 1]
        v_u = upper[..., 1]

        v_abs_max = torch.maximum(v_l.abs(), v_u.abs())
        kinetic_l = -0.5 * v_abs_max.pow(2)
        crosses_zero = (v_l <= 0) & (v_u >= 0)
        kinetic_u = torch.where(crosses_zero, torch.zeros_like(v_l), -0.5 * torch.minimum(v_l.pow(2), v_u.pow(2)))

        # sin(x) = cos(x - pi/2)
        sin_l, sin_u = cos_interval(3.0 * p_l - torch.pi / 2.0, 3.0 * p_u - torch.pi / 2.0)
        potential_l, potential_u = scalar_mul_interval(sin_l, sin_u, -(0.0025 / 3.0))
        slope_l = 0.0015 * p_l
        slope_u = 0.0015 * p_u
        raw_l, raw_u = add_interval(kinetic_l, kinetic_u, potential_l, potential_u)
        raw_l, raw_u = add_interval(raw_l, raw_u, slope_l, slope_u)
        c = self.c.to(dtype=lower.dtype, device=lower.device)
        return raw_l + c, raw_u + c

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"type": "energy", "p_safe": self.p_safe, "c": self.c.detach().cpu()}, path)
