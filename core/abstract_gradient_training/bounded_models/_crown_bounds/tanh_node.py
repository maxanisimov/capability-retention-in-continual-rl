"""CROWN node and relaxation helpers for tanh activations."""

from __future__ import annotations

import logging
from typing import Literal

import torch

from abstract_gradient_training.bounded_models._crown_bounds import (
    Node,
    LinearBounds,
    IntervalBounds,
)

LOGGER = logging.getLogger(__name__)
_MIXED_TANH_WARNING_EMITTED = False


def _effective_eps(x: torch.Tensor, eps: float) -> float:
    return max(float(eps), float(torch.finfo(x.dtype).eps))


def _atanh_clamped(x: torch.Tensor, eps: float) -> torch.Tensor:
    eps = _effective_eps(x, eps)
    x = x.clamp(min=-1 + eps, max=1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _tanh_derivative(x: torch.Tensor) -> torch.Tensor:
    return 1 - torch.tanh(x).square()


def _raw_to_slope(raw: torch.Tensor, eps: float) -> torch.Tensor:
    eps = _effective_eps(raw, eps)
    return eps + (1 - 2 * eps) * torch.sigmoid(raw)


def _slope_to_raw(slope: torch.Tensor, eps: float) -> torch.Tensor:
    eps = _effective_eps(slope, eps)
    slope = slope.clamp(min=eps, max=1 - eps)
    unit_slope = (slope - eps) / (1 - 2 * eps)
    unit_slope = unit_slope.clamp(min=eps, max=1 - eps)
    return torch.logit(unit_slope)


def _residual_bounds(
    l: torch.Tensor, u: torch.Tensor, slope: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact extrema of tanh(z) - slope * z over [l, u]."""
    eps = _effective_eps(l, eps)
    slope_clamped = slope.clamp(min=eps, max=1 - eps)
    root = torch.sqrt((1 - slope_clamped).clamp(min=0))
    crit_pos = _atanh_clamped(root, eps)
    crit_neg = -crit_pos

    endpoint_l = torch.tanh(l) - slope * l
    endpoint_u = torch.tanh(u) - slope * u
    residuals_for_min = [endpoint_l, endpoint_u]
    residuals_for_max = [endpoint_l, endpoint_u]
    inf = torch.full_like(l, float("inf"))
    neg_inf = torch.full_like(l, -float("inf"))
    for point in (crit_neg, crit_pos):
        residual = torch.tanh(point) - slope * point
        inside = (l <= point) & (point <= u)
        residuals_for_min.append(torch.where(inside, residual, inf))
        residuals_for_max.append(torch.where(inside, residual, neg_inf))

    return (
        torch.stack(residuals_for_min).min(dim=0).values,
        torch.stack(residuals_for_max).max(dim=0).values,
    )


def tanh_linear_bounds(
    l: torch.Tensor, u: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute affine lower and upper bounds for tanh over ``[l, u]``.

    Returns coefficients ``a_l, b_l, a_u, b_u`` such that
    ``a_l * z + b_l <= tanh(z) <= a_u * z + b_u`` for all ``z`` in the
    interval. One-sided intervals use curvature-aware secant/tangent bounds.
    Mixed intervals use a sound global residual envelope because this CROWN
    graph does not support splitting one neuron into multiple domains.
    """
    global _MIXED_TANH_WARNING_EMITTED
    l, u = torch.broadcast_tensors(l, u)
    eps = _effective_eps(l, eps)
    width = u - l
    degenerate = width == 0
    safe_width = torch.where(degenerate, torch.ones_like(width), width)
    tanh_l = torch.tanh(l)
    tanh_u = torch.tanh(u)
    secant = (tanh_u - tanh_l) / safe_width

    # Conservative global fallback: fixed slope plus exact residual extrema.
    residual_l, residual_u = _residual_bounds(l, u, secant, eps)
    a_l = secant.clone()
    b_l = residual_l
    a_u = secant.clone()
    b_u = residual_u

    secant_clamped = secant.clamp(min=eps, max=1 - eps)
    root = torch.sqrt((1 - secant_clamped).clamp(min=0))

    # Entirely negative: tanh is convex, so tangent <= tanh <= secant.
    neg = (u <= 0) & ~degenerate
    d_neg = _atanh_clamped(-root, eps)
    tanh_d_neg = torch.tanh(d_neg)
    tangent_l_neg = secant_clamped
    tangent_b_neg = tanh_d_neg - tangent_l_neg * d_neg
    secant_b = tanh_l - secant * l
    a_l = torch.where(neg, tangent_l_neg, a_l)
    b_l = torch.where(neg, tangent_b_neg, b_l)
    a_u = torch.where(neg, secant, a_u)
    b_u = torch.where(neg, secant_b, b_u)

    # Entirely positive: tanh is concave, so secant <= tanh <= tangent.
    pos = (l >= 0) & ~degenerate
    d_pos = _atanh_clamped(root, eps)
    tanh_d_pos = torch.tanh(d_pos)
    tangent_u_pos = secant_clamped
    tangent_b_pos = tanh_d_pos - tangent_u_pos * d_pos
    a_l = torch.where(pos, secant, a_l)
    b_l = torch.where(pos, secant_b, b_l)
    a_u = torch.where(pos, tangent_u_pos, a_u)
    b_u = torch.where(pos, tangent_b_pos, b_u)

    # Degenerate intervals use the exact local linearisation.
    deriv = _tanh_derivative(l)
    exact_b = tanh_l - deriv * l
    a_l = torch.where(degenerate, deriv, a_l)
    b_l = torch.where(degenerate, exact_b, b_l)
    a_u = torch.where(degenerate, deriv, a_u)
    b_u = torch.where(degenerate, exact_b, b_u)

    mixed = (l < 0) & (u > 0) & ~degenerate
    if bool(mixed.any()) and not _MIXED_TANH_WARNING_EMITTED:
        LOGGER.warning(
            "Tanh CROWN bounds encountered mixed-sign intervals. Using a sound "
            "single-affine residual fallback; bounds may be looser than split-domain bounds."
        )
        _MIXED_TANH_WARNING_EMITTED = True

    if not all(torch.isfinite(t).all() for t in (a_l, b_l, a_u, b_u)):
        raise ValueError("Tanh linear relaxation produced non-finite coefficients.")

    return a_l, b_l, a_u, b_u


class TanhNode(Node):
    """
    A CROWN node representing out_var = tanh(in_var).

    The node stores diagonal affine envelopes
        alpha_l * x + beta_l <= tanh(x) <= alpha_u * x + beta_u
    and chooses the appropriate envelope during backpropagation based on the
    sign of the propagated coefficient.
    """

    def __init__(
        self,
        in_var: Node,
        interval_matmul: Literal["rump", "exact", "nguyen"] = "rump",
        tanh_relaxation: Literal["fixed", "optimizable"] = "fixed",
    ):
        super().__init__()
        assert interval_matmul in [
            "rump",
            "exact",
            "nguyen",
        ], f"Unknown interval matmul method: {interval_matmul}"
        assert tanh_relaxation in [
            "fixed",
            "optimizable",
        ], f"Unknown tanh relaxation method: {tanh_relaxation}"
        self.in_var = in_var
        self.interval_matmul: Literal["rump", "exact", "nguyen"] = interval_matmul
        self.tanh_relaxation = tanh_relaxation
        self.eps = 1e-12
        self.initialized = False
        self.update_relaxation()

    def update_relaxation(self) -> None:
        x_l, x_u = self.in_var.concretize()  # type: ignore
        assert x_l.shape == x_u.shape
        assert x_l.dim() == 2, "Expected input to be a 2D tensor"
        fixed_alpha_l, fixed_beta_l, fixed_alpha_u, fixed_beta_u = tanh_linear_bounds(
            x_l, x_u, eps=self.eps
        )

        if self.tanh_relaxation == "fixed":
            self.alpha_l, self.beta_l = fixed_alpha_l, fixed_beta_l
            self.alpha_u, self.beta_u = fixed_alpha_u, fixed_beta_u
            self.initialized = True
            return

        if not self.initialized:
            self.alpha_l_raw = _slope_to_raw(
                fixed_alpha_l, self.eps
            ).detach().requires_grad_()
            self.alpha_u_raw = _slope_to_raw(
                fixed_alpha_u, self.eps
            ).detach().requires_grad_()
            self._optimizable_params.extend([self.alpha_l_raw, self.alpha_u_raw])

        alpha_l = _raw_to_slope(self.alpha_l_raw, self.eps)
        alpha_u = _raw_to_slope(self.alpha_u_raw, self.eps)
        beta_l, _ = _residual_bounds(x_l, x_u, alpha_l, self.eps)
        _, beta_u = _residual_bounds(x_l, x_u, alpha_u, self.eps)

        # For exact point intervals, keep the exact local linearisation.
        degenerate = x_l == x_u
        deriv = _tanh_derivative(x_l)
        exact_beta = torch.tanh(x_l) - deriv * x_l
        self.alpha_l = torch.where(degenerate, deriv, alpha_l)
        self.beta_l = torch.where(degenerate, exact_beta, beta_l)
        self.alpha_u = torch.where(degenerate, deriv, alpha_u)
        self.beta_u = torch.where(degenerate, exact_beta, beta_u)
        self.initialized = True

    def _backpropagate(self, backward_bounds: LinearBounds) -> None:
        self.update_relaxation()
        Lambda, Omega, delta, theta = backward_bounds
        conc = self.concretize()

        # Upper-bound propagation: positive coefficients use the upper envelope,
        # negative coefficients use the lower envelope.
        pos_mask = Lambda.lb >= 0
        neg_mask = Lambda.ub <= 0
        mixed_mask = (Lambda.lb < 0) & (Lambda.ub > 0)
        backward_bounds.delta = (
            delta
            + (Lambda * self.beta_u.unsqueeze(1) * pos_mask).sum(-1)
            + (Lambda * self.beta_l.unsqueeze(1) * neg_mask).sum(-1)
            + ((Lambda * mixed_mask) @ conc.unsqueeze(-1)).squeeze(-1)
        )
        backward_bounds.Lambda = Lambda * (
            self.alpha_u.unsqueeze(1) * pos_mask
            + self.alpha_l.unsqueeze(1) * neg_mask
        )

        # Lower-bound propagation: positive coefficients use the lower envelope,
        # negative coefficients use the upper envelope.
        pos_mask = Omega.lb >= 0
        neg_mask = Omega.ub <= 0
        mixed_mask = (Omega.lb < 0) & (Omega.ub > 0)
        backward_bounds.theta = (
            theta
            + (Omega * self.beta_l.unsqueeze(1) * pos_mask).sum(-1)
            + (Omega * self.beta_u.unsqueeze(1) * neg_mask).sum(-1)
            + ((Omega * mixed_mask) @ conc.unsqueeze(-1)).squeeze(-1)
        )
        backward_bounds.Omega = Omega * (
            self.alpha_l.unsqueeze(1) * pos_mask
            + self.alpha_u.unsqueeze(1) * neg_mask
        )

    def _init_backpropagation(self) -> LinearBounds:
        self.update_relaxation()
        Lambda0 = IntervalBounds(
            torch.diag_embed(self.alpha_u.flatten(start_dim=1), dim1=-2, dim2=-1),
            interval_matmul=self.interval_matmul,
        )
        Omega0 = IntervalBounds(
            torch.diag_embed(self.alpha_l.flatten(start_dim=1), dim1=-2, dim2=-1),
            interval_matmul=self.interval_matmul,
        )
        delta0 = IntervalBounds(self.beta_u.clone(), interval_matmul=self.interval_matmul)
        theta0 = IntervalBounds(self.beta_l.clone(), interval_matmul=self.interval_matmul)
        return LinearBounds(Lambda0, Omega0, delta0, theta0)
