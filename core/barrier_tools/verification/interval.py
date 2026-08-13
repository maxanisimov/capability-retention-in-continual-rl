"""Small interval-arithmetic helpers used by the barrier verifier."""

from __future__ import annotations

import math

import torch


def validate_interval(lower: torch.Tensor, upper: torch.Tensor, *, name: str = "interval") -> None:
    """Raise if any lower bound exceeds its upper bound."""

    if bool((lower > upper).any().item()):
        raise ValueError(f"Invalid {name}: lower bound exceeds upper bound.")


def affine_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound ``x @ weight.T + bias`` for an input interval."""

    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)
    out_lower = lower @ weight_pos.T + upper @ weight_neg.T
    out_upper = upper @ weight_pos.T + lower @ weight_neg.T
    if bias is not None:
        out_lower = out_lower + bias
        out_upper = out_upper + bias
    return out_lower, out_upper


def mul_interval(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounds for the product of two intervals."""

    candidates = torch.stack(
        [
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        ],
        dim=0,
    )
    return candidates.min(dim=0).values, candidates.max(dim=0).values


def scalar_mul_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    scalar: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounds for a scalar times an interval."""

    scalar_t = torch.as_tensor(scalar, dtype=lower.dtype, device=lower.device)
    lo = lower * scalar_t
    hi = upper * scalar_t
    return torch.minimum(lo, hi), torch.maximum(lo, hi)


def add_interval(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounds for the sum of two intervals."""

    return left_lower + right_lower, left_upper + right_upper


def clip_interval(
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_value: float,
    max_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounds after clipping an interval to ``[min_value, max_value]``."""

    return torch.clamp(lower, min_value, max_value), torch.clamp(upper, min_value, max_value)


def pow_interval(lower: torch.Tensor, upper: torch.Tensor, exponent: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounds for an integer power over an interval."""

    if exponent < 0:
        raise ValueError("Only non-negative integer powers are supported.")
    if exponent == 0:
        ones = torch.ones_like(lower)
        return ones, ones
    if exponent == 1:
        return lower, upper

    lower_p = lower.pow(exponent)
    upper_p = upper.pow(exponent)
    if exponent % 2 == 1:
        return lower_p, upper_p

    crosses_zero = (lower <= 0) & (upper >= 0)
    out_lower = torch.where(crosses_zero, torch.zeros_like(lower), torch.minimum(lower_p, upper_p))
    out_upper = torch.maximum(lower_p, upper_p)
    return out_lower, out_upper


def cos_interval(lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sound bounds for ``cos(x)`` over each scalar interval in a tensor."""

    flat_lower = lower.reshape(-1)
    flat_upper = upper.reshape(-1)
    lo_values: list[float] = []
    hi_values: list[float] = []
    two_pi = 2.0 * math.pi
    for lo_t, hi_t in zip(flat_lower, flat_upper):
        lo = float(lo_t.item())
        hi = float(hi_t.item())
        if hi < lo:
            raise ValueError("Invalid cosine interval.")
        if hi - lo >= two_pi:
            lo_values.append(-1.0)
            hi_values.append(1.0)
            continue

        candidates = [math.cos(lo), math.cos(hi)]
        first = math.ceil(lo / math.pi)
        last = math.floor(hi / math.pi)
        for k in range(first, last + 1):
            candidates.append(math.cos(k * math.pi))
        lo_values.append(min(candidates))
        hi_values.append(max(candidates))

    cos_l = torch.tensor(lo_values, dtype=lower.dtype, device=lower.device).reshape(lower.shape)
    cos_u = torch.tensor(hi_values, dtype=lower.dtype, device=lower.device).reshape(lower.shape)
    return cos_l, cos_u
