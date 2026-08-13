"""Trainable low-degree polynomial barriers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from barrier_tools.barriers.energy_barrier import EnergyBarrier
from barrier_tools.verification.interval import (
    mul_interval,
    pow_interval,
    scalar_mul_interval,
)


def monomial_powers(degree: int, dim: int = 2) -> list[tuple[int, ...]]:
    """Return all monomial powers with total degree at most ``degree``."""

    if dim != 2:
        raise ValueError("Only two-dimensional polynomials are supported in v1.")
    powers: list[tuple[int, ...]] = []
    for total in range(int(degree) + 1):
        for i in range(total + 1):
            powers.append((i, total - i))
    return powers


class PolynomialBarrier(nn.Module):
    """A polynomial barrier, optionally added to the analytic energy candidate."""

    def __init__(
        self,
        degree: int = 4,
        *,
        use_energy_base: bool = True,
        p_safe: float = -1.0,
        coefficients: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.degree = int(degree)
        self.powers = monomial_powers(self.degree)
        self.use_energy_base = bool(use_energy_base)
        self.energy_base = EnergyBarrier(p_safe=p_safe) if use_energy_base else None
        initial = torch.zeros(len(self.powers), dtype=torch.float32)
        if coefficients is not None:
            initial = coefficients.detach().clone().float()
        self.coefficients = nn.Parameter(initial)

    def correction(self, state: torch.Tensor) -> torch.Tensor:
        values = []
        p = state[..., 0]
        v = state[..., 1]
        for p_power, v_power in self.powers:
            values.append(p.pow(p_power) * v.pow(v_power))
        basis = torch.stack(values, dim=-1)
        return basis @ self.coefficients.to(dtype=state.dtype, device=state.device)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        out = self.correction(state)
        if self.energy_base is not None:
            out = out + self.energy_base.value(state)
        return out

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(state)

    def interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_l = lower[..., 0]
        p_u = upper[..., 0]
        v_l = lower[..., 1]
        v_u = upper[..., 1]
        total_l = torch.zeros_like(p_l)
        total_u = torch.zeros_like(p_l)
        coeffs = self.coefficients.to(dtype=lower.dtype, device=lower.device)
        for coeff, (p_power, v_power) in zip(coeffs, self.powers):
            p_term_l, p_term_u = pow_interval(p_l, p_u, p_power)
            v_term_l, v_term_u = pow_interval(v_l, v_u, v_power)
            term_l, term_u = mul_interval(p_term_l, p_term_u, v_term_l, v_term_u)
            term_l, term_u = scalar_mul_interval(term_l, term_u, coeff)
            total_l = total_l + term_l
            total_u = total_u + term_u

        if self.energy_base is not None:
            base_l, base_u = self.energy_base.interval(lower, upper)
            total_l = total_l + base_l
            total_u = total_u + base_u
        return total_l, total_u

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "type": "polynomial",
                "degree": self.degree,
                "powers": self.powers,
                "use_energy_base": self.use_energy_base,
                "coefficients": self.coefficients.detach().cpu(),
            },
            path,
        )
