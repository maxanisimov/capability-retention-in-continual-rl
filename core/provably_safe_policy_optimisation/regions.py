"""Safe parameter-region representations and projection helpers.

The historical PSPO region is an axis-aligned parameter box.  Zonotope regions
share the same projection interface but represent correlated parameter motion as
``theta = center + z @ generators`` with bounded coefficients ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from provably_safe_policy_optimisation.projection import (
    ProjectionResult,
    project_to_interval_union,
)

SafeRegionShape = Literal["orthotope", "zonotope"]


@dataclass(frozen=True)
class OrthotopeRegion:
    """One axis-aligned parameter box."""

    lower: list[torch.Tensor]
    upper: list[torch.Tensor]
    shape: SafeRegionShape = "orthotope"


@dataclass(frozen=True)
class ZonotopeRegion:
    """One low-rank zonotope in flattened parameter space."""

    center_params: list[torch.Tensor]
    generators: torch.Tensor
    coefficient_l: torch.Tensor
    coefficient_u: torch.Tensor
    param_shapes: list[tuple[int, ...]]
    shape: SafeRegionShape = "zonotope"


SafeParameterRegion = OrthotopeRegion | ZonotopeRegion


def flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Flatten and concatenate tensors in their existing order."""

    if not tensors:
        return torch.empty(0)
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def unflatten_like(flat: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split a flat tensor into tensors shaped like ``like``."""

    out: list[torch.Tensor] = []
    offset = 0
    for tensor in like:
        n = tensor.numel()
        out.append(flat[offset : offset + n].reshape_as(tensor))
        offset += n
    if offset != flat.numel():
        raise ValueError(
            f"Flat tensor has {flat.numel()} entries but target shapes consume {offset}."
        )
    return out


def orthotope_from_bounds(lower: list[torch.Tensor], upper: list[torch.Tensor]) -> OrthotopeRegion:
    """Build an orthotope region from one lower/upper tensor list."""

    return OrthotopeRegion(
        lower=[tensor.detach().clone() for tensor in lower],
        upper=[tensor.detach().clone() for tensor in upper],
    )


def zonotope_rank_default(n_params: int) -> int:
    """Default zonotope rank used when the user does not specify one."""

    return max(1, min(16, int(n_params)))


def validate_zonotope_region(region: ZonotopeRegion, params: list[torch.Tensor]) -> None:
    """Validate a zonotope region against a parameter order."""

    if len(region.center_params) != len(params):
        raise ValueError(
            "Zonotope center must provide one tensor per projected parameter: "
            f"expected={len(params)}, got={len(region.center_params)}.",
        )
    for idx, (center, param) in enumerate(zip(region.center_params, params)):
        if tuple(center.shape) != tuple(param.shape):
            raise ValueError(
                f"Zonotope center shape mismatch at param {idx}: "
                f"expected={tuple(param.shape)}, got={tuple(center.shape)}.",
            )
    if len(region.param_shapes) != len(params):
        raise ValueError(
            "Zonotope param_shapes must provide one shape per projected parameter: "
            f"expected={len(params)}, got={len(region.param_shapes)}.",
        )
    for idx, (recorded_shape, param) in enumerate(zip(region.param_shapes, params)):
        if tuple(recorded_shape) != tuple(param.shape):
            raise ValueError(
                f"Zonotope param_shapes mismatch at param {idx}: "
                f"expected={tuple(param.shape)}, got={tuple(recorded_shape)}.",
            )
    n_params = sum(param.numel() for param in params)
    if region.generators.ndim != 2 or int(region.generators.shape[1]) != n_params:
        raise ValueError(
            "Zonotope generators must have shape (rank, n_params): "
            f"expected second dimension {n_params}, got {tuple(region.generators.shape)}.",
        )
    rank = int(region.generators.shape[0])
    if tuple(region.coefficient_l.shape) != (rank,) or tuple(region.coefficient_u.shape) != (rank,):
        raise ValueError(
            "Zonotope coefficient bounds must be length-rank tensors: "
            f"rank={rank}, lower={tuple(region.coefficient_l.shape)}, "
            f"upper={tuple(region.coefficient_u.shape)}.",
        )
    if not bool(torch.all(region.coefficient_l <= region.coefficient_u)):
        raise ValueError("Zonotope coefficient lower bounds must be <= upper bounds.")


def prepare_region_for_params(
    region: SafeParameterRegion,
    params: list[torch.nn.Parameter],
    *,
    device: torch.device,
) -> SafeParameterRegion:
    """Move a region onto the device/dtypes of a parameter order."""

    dtypes = [param.dtype for param in params]
    if isinstance(region, OrthotopeRegion):
        if len(region.lower) != len(params) or len(region.upper) != len(params):
            raise ValueError(
                "Orthotope region must provide one lower/upper tensor per parameter."
            )
        lower: list[torch.Tensor] = []
        upper: list[torch.Tensor] = []
        for idx, (lb, ub, param) in enumerate(zip(region.lower, region.upper, params)):
            if tuple(lb.shape) != tuple(param.shape) or tuple(ub.shape) != tuple(param.shape):
                raise ValueError(
                    f"Orthotope shape mismatch at param {idx}: param={tuple(param.shape)}, "
                    f"lower={tuple(lb.shape)}, upper={tuple(ub.shape)}.",
                )
            lb_t = lb.to(device=device, dtype=dtypes[idx])
            ub_t = ub.to(device=device, dtype=dtypes[idx])
            if not bool(torch.all(lb_t <= ub_t)):
                raise ValueError(f"Empty orthotope at param {idx}: lower exceeds upper.")
            lower.append(lb_t)
            upper.append(ub_t)
        return OrthotopeRegion(lower=lower, upper=upper)

    centers = [
        center.to(device=device, dtype=dtypes[idx])
        for idx, center in enumerate(region.center_params)
    ]
    prepared = ZonotopeRegion(
        center_params=centers,
        generators=region.generators.to(device=device, dtype=dtypes[0]),
        coefficient_l=region.coefficient_l.to(device=device, dtype=dtypes[0]),
        coefficient_u=region.coefficient_u.to(device=device, dtype=dtypes[0]),
        param_shapes=[tuple(param.shape) for param in params],
    )
    validate_zonotope_region(prepared, list(params))
    return prepared


def zonotope_flat_center(region: ZonotopeRegion) -> torch.Tensor:
    """Return a flat zonotope center vector."""

    return flatten_tensors(region.center_params).to(
        device=region.generators.device,
        dtype=region.generators.dtype,
    )


def project_flat_to_zonotope(
    flat_params: torch.Tensor,
    region: ZonotopeRegion,
    *,
    n_iters: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a flat parameter vector onto a zonotope.

    Returns ``(projected_flat_params, coefficients)``.  The solve starts with a
    least-squares coefficient estimate and refines it with projected gradient
    descent over the bounded coefficient vector.
    """

    center = zonotope_flat_center(region).to(device=flat_params.device, dtype=flat_params.dtype)
    generators = region.generators.to(device=flat_params.device, dtype=flat_params.dtype)
    coeff_l = region.coefficient_l.to(device=flat_params.device, dtype=flat_params.dtype)
    coeff_u = region.coefficient_u.to(device=flat_params.device, dtype=flat_params.dtype)
    if generators.numel() == 0:
        return center, torch.empty(0, device=flat_params.device, dtype=flat_params.dtype)

    delta = flat_params - center
    # Solve generators.T @ z ~= delta.
    try:
        z0 = torch.linalg.lstsq(generators.T, delta.unsqueeze(1)).solution.squeeze(1)
    except RuntimeError:
        z0 = torch.zeros(generators.shape[0], device=flat_params.device, dtype=flat_params.dtype)
    z = z0.clamp(min=coeff_l, max=coeff_u).detach().clone().requires_grad_(True)
    candidate = center + z.detach() @ generators
    if bool(torch.allclose(candidate, flat_params, atol=1e-7, rtol=1e-6)):
        return flat_params.detach().clone(), z.detach()

    lr = 0.5 / (float(torch.linalg.matrix_norm(generators).item()) ** 2 + 1e-6)
    with torch.enable_grad():
        for _ in range(int(n_iters)):
            projected = center + z @ generators
            loss = torch.sum((projected - flat_params.detach()) ** 2)
            grad = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                z -= lr * grad
                z.clamp_(min=coeff_l, max=coeff_u)
    z_final = z.detach()
    return center + z_final @ generators, z_final


def project_to_region_union(
    actor_params: list[torch.nn.Parameter],
    regions: list[SafeParameterRegion],
    *,
    distance_norm: str = "l2",
    apply: bool = True,
) -> ProjectionResult:
    """Project parameters onto the nearest region from a box/zonotope union."""

    if not regions:
        raise ValueError("At least one safe parameter region is required.")
    norm = str(distance_norm).strip().lower()
    if norm in {"l_inf", "inf", "infty", "infinity"}:
        norm = "linf"
    if norm not in {"l2", "l1", "linf"}:
        raise ValueError(
            f"Unsupported projection distance norm {distance_norm!r}; expected l2, l1, or linf."
        )

    best_idx: int | None = None
    best_distance = float("inf")
    best_projected: list[torch.Tensor] | None = None
    best_result: ProjectionResult | None = None

    for idx, region in enumerate(regions):
        if isinstance(region, OrthotopeRegion):
            shadow = [torch.nn.Parameter(param.detach().clone()) for param in actor_params]
            result = project_to_interval_union(
                shadow,
                [region.lower],
                [region.upper],
                distance_norm=norm,
            )
            projected = [param.detach().clone() for param in shadow]
        else:
            flat = flatten_tensors([param.detach() for param in actor_params])
            projected_flat, coeffs = project_flat_to_zonotope(flat, region)
            projected = unflatten_like(projected_flat, [param.detach() for param in actor_params])
            deltas = [proj - param.detach() for proj, param in zip(projected, actor_params)]
            n_projected = int(
                sum(int(torch.count_nonzero(delta != 0.0).item()) for delta in deltas)
            )
            coeff_l = region.coefficient_l.to(device=coeffs.device, dtype=coeffs.dtype)
            coeff_u = region.coefficient_u.to(device=coeffs.device, dtype=coeffs.dtype)
            n_boundary = int(
                (torch.isclose(coeffs, coeff_l) | torch.isclose(coeffs, coeff_u)).sum().item()
            )
            l2_sq = sum(float(torch.sum(delta * delta).item()) for delta in deltas)
            linf = max(
                (float(torch.max(torch.abs(delta)).item()) for delta in deltas if delta.numel()),
                default=0.0,
            )
            result = ProjectionResult(
                n_projected=n_projected,
                n_boundary=n_boundary,
                selected_set_index=idx,
                displacement_l2=float(l2_sq ** 0.5),
                displacement_linf=float(linf),
            )

        if norm == "l2":
            distance = result.displacement_l2 ** 2
        elif norm == "l1":
            distance = sum(
                float(torch.sum(torch.abs(proj - param.detach())).item())
                for proj, param in zip(projected, actor_params)
            )
        else:
            distance = result.displacement_linf
        if distance < best_distance:
            best_idx = idx
            best_distance = distance
            best_projected = projected
            best_result = result
            if distance == 0.0:
                break

    if best_idx is None or best_projected is None or best_result is None:
        raise RuntimeError("Failed to select a safe parameter region for projection.")
    if apply:
        for param, projected in zip(actor_params, best_projected):
            param.data.copy_(projected.to(device=param.device, dtype=param.dtype))
    return ProjectionResult(
        n_projected=best_result.n_projected,
        n_boundary=best_result.n_boundary,
        selected_set_index=int(best_idx),
        displacement_l2=best_result.displacement_l2,
        displacement_linf=best_result.displacement_linf,
    )
