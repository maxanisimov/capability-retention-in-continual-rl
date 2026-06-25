"""Parameter-space projection primitives for provably-safe policy optimisation.

These helpers project a network's parameters onto a *union of axis-aligned
boxes* (a "Rashomon set"): each box is a per-parameter interval ``[lower, upper]``
and projection snaps the parameters into the nearest box. They are the shared
core of projected gradient descent used by both the custom PPO trainer
(``experiments.utils.ppo_utils``) and the Stable-Baselines3 wrappers
(:class:`~provably_safe_policy_optimisation.projected_optimizers.ProjectedAdam`).

Bounds layouts accepted by :func:`validate_and_prepare_param_interval_bounds`:

* single box: ``list[Tensor]`` (one lower/upper tensor per parameter), or
* union of boxes: ``list[list[Tensor]]`` (interval-major or parameter-major).

All are normalised to *set-major* form ``bounds[set_idx][param_idx] -> Tensor``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# A single box (list[Tensor], one per parameter) or a union of boxes
# (list[list[Tensor]], interval-major or parameter-major).
ActorParamBounds = list[torch.Tensor] | list[list[torch.Tensor]]


@dataclass(frozen=True)
class ProjectionResult:
    """Diagnostics for a single :func:`project_to_interval_union` call.

    Attributes
    ----------
    n_projected:
        Number of constrained scalar entries that were strictly outside the
        selected box and therefore clamped this step.
    n_boundary:
        Number of constrained scalar entries lying exactly on a box face after
        projection (active constraints / "pinned" weights).
    selected_set_index:
        Index of the box (Rashomon set) the parameters were projected onto.
    displacement_l2:
        Euclidean norm of the projection step over all constrained parameters
        (how far the parameters were pushed back into bounds). ``0.0`` when no
        clamping was needed.
    displacement_linf:
        Largest absolute single-coordinate move of the projection step.
    """

    n_projected: int
    n_boundary: int
    selected_set_index: int
    displacement_l2: float
    displacement_linf: float


def validate_and_prepare_param_interval_bounds(
    *,
    actor_params: list[torch.nn.Parameter],
    actor_param_bounds_l: ActorParamBounds,
    actor_param_bounds_u: ActorParamBounds,
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """
    Normalize PGD bounds into set-major interval lists.

    Accepted formats:
      1) Single interval (backward-compatible):
         actor_param_bounds_l/u: list[Tensor] with one lower/upper tensor per parameter.
      2) Multiple intervals:
         - interval-major: list[list[Tensor]] where outer index is interval and
           inner index is parameter.
         - parameter-major: list[list[Tensor]] where outer index is parameter and
           inner index is interval.

    Returns:
      (bounds_l_sets, bounds_u_sets) where each is set-major:
        bounds_*_sets[set_idx][param_idx] -> Tensor
    """
    n_params = len(actor_params)

    def _is_tensor_list(x: object) -> bool:
        return isinstance(x, list) and all(isinstance(v, torch.Tensor) for v in x)

    def _is_nested_tensor_list(x: object) -> bool:
        return (
            isinstance(x, list)
            and all(isinstance(v, list) for v in x)
            and all(all(isinstance(t, torch.Tensor) for t in v) for v in x)
        )

    actor_shapes = [tuple(p.shape) for p in actor_params]

    # Single-interval format: list[Tensor], list[Tensor]
    if _is_tensor_list(actor_param_bounds_l) and _is_tensor_list(actor_param_bounds_u):
        if len(actor_param_bounds_l) != n_params or len(actor_param_bounds_u) != n_params:
            raise ValueError(
                "Single-interval PGD bounds must provide one tensor per actor parameter. "
                f"Expected {n_params}, got lower={len(actor_param_bounds_l)} upper={len(actor_param_bounds_u)}.",
            )

        set_l: list[torch.Tensor] = []
        set_u: list[torch.Tensor] = []
        for p_idx, (lb, ub, expected_shape) in enumerate(
            zip(actor_param_bounds_l, actor_param_bounds_u, actor_shapes),
        ):
            if tuple(lb.shape) != expected_shape or tuple(ub.shape) != expected_shape:
                raise ValueError(
                    f"PGD bound shape mismatch at param index {p_idx}: "
                    f"expected={expected_shape}, lower={tuple(lb.shape)}, upper={tuple(ub.shape)}",
                )
            set_l.append(lb.to(device))
            set_u.append(ub.to(device))
        return [set_l], [set_u]

    # Multi-interval formats: nested lists.
    if not (_is_nested_tensor_list(actor_param_bounds_l) and _is_nested_tensor_list(actor_param_bounds_u)):
        raise TypeError(
            "PGD bounds must be either list[Tensor] (single interval) or list[list[Tensor]] "
            "(multiple intervals).",
        )

    if len(actor_param_bounds_l) != len(actor_param_bounds_u):
        raise ValueError(
            "Lower/upper nested PGD bounds outer lengths do not match: "
            f"lower={len(actor_param_bounds_l)}, upper={len(actor_param_bounds_u)}.",
        )
    if len(actor_param_bounds_l) == 0:
        raise ValueError("Nested PGD bounds must contain at least one interval/parameter group.")

    def _is_valid_interval_major(
        nested_l: list[list[torch.Tensor]],
        nested_u: list[list[torch.Tensor]],
    ) -> bool:
        for int_l, int_u in zip(nested_l, nested_u):
            if len(int_l) != n_params or len(int_u) != n_params:
                return False
            for p_idx, expected_shape in enumerate(actor_shapes):
                if tuple(int_l[p_idx].shape) != expected_shape or tuple(int_u[p_idx].shape) != expected_shape:
                    return False
        return True

    def _is_valid_parameter_major(
        nested_l: list[list[torch.Tensor]],
        nested_u: list[list[torch.Tensor]],
    ) -> bool:
        if len(nested_l) != n_params or len(nested_u) != n_params:
            return False
        if len(nested_l[0]) == 0:
            return False
        n_intervals = len(nested_l[0])
        for p_idx, (param_l, param_u, expected_shape) in enumerate(
            zip(nested_l, nested_u, actor_shapes),
        ):
            if len(param_l) != n_intervals or len(param_u) != n_intervals:
                return False
            for int_idx in range(n_intervals):
                if tuple(param_l[int_idx].shape) != expected_shape or tuple(param_u[int_idx].shape) != expected_shape:
                    return False
        return True

    interval_major_ok = _is_valid_interval_major(actor_param_bounds_l, actor_param_bounds_u)
    parameter_major_ok = _is_valid_parameter_major(actor_param_bounds_l, actor_param_bounds_u)

    # Prefer interval-major when both are syntactically possible.
    if interval_major_ok:
        bounds_l_sets: list[list[torch.Tensor]] = []
        bounds_u_sets: list[list[torch.Tensor]] = []
        for int_l, int_u in zip(actor_param_bounds_l, actor_param_bounds_u):
            bounds_l_sets.append([t.to(device) for t in int_l])
            bounds_u_sets.append([t.to(device) for t in int_u])
        return bounds_l_sets, bounds_u_sets

    if parameter_major_ok:
        n_intervals = len(actor_param_bounds_l[0])
        bounds_l_sets = []
        bounds_u_sets = []
        for int_idx in range(n_intervals):
            set_l = [actor_param_bounds_l[p_idx][int_idx].to(device) for p_idx in range(n_params)]
            set_u = [actor_param_bounds_u[p_idx][int_idx].to(device) for p_idx in range(n_params)]
            bounds_l_sets.append(set_l)
            bounds_u_sets.append(set_u)
        return bounds_l_sets, bounds_u_sets

    raise ValueError(
        "Nested PGD bounds do not match either supported layout:\n"
        "- interval-major: bounds[interval][parameter]\n"
        "- parameter-major: bounds[parameter][interval]",
    )


def project_to_interval_union(
    actor_params: list[torch.nn.Parameter],
    bounds_l_sets: list[list[torch.Tensor]],
    bounds_u_sets: list[list[torch.Tensor]],
    distance_norm: str = "l2",
) -> ProjectionResult:
    """
    Project actor parameters onto the nearest convex Rashomon set (box) in full parameter space.

    IMPORTANT: This preserves set coupling across parameters.

    Returns a :class:`ProjectionResult` with per-step projection diagnostics.
    """
    norm = str(distance_norm).strip().lower()
    if norm in {"l_inf", "inf", "infty", "infinity"}:
        norm = "linf"
    if norm not in {"l2", "l1", "linf"}:
        raise ValueError(
            "Unsupported projection distance norm. "
            f"Got '{distance_norm}', expected one of: 'l2', 'l1', 'linf'.",
        )

    if len(bounds_l_sets) != len(bounds_u_sets):
        raise ValueError(
            "Set-major lower/upper bounds length mismatch: "
            f"lower_sets={len(bounds_l_sets)}, upper_sets={len(bounds_u_sets)}",
        )
    if len(bounds_l_sets) == 0:
        raise ValueError("At least one convex Rashomon set is required for PGD projection.")

    n_params = len(actor_params)
    for set_idx, (set_l, set_u) in enumerate(zip(bounds_l_sets, bounds_u_sets)):
        if len(set_l) != n_params or len(set_u) != n_params:
            raise ValueError(
                f"Rashomon set {set_idx} must provide bounds for all parameters: "
                f"expected={n_params}, lower={len(set_l)}, upper={len(set_u)}",
            )

    best_set_idx: int | None = None
    best_distance = float("inf")
    best_projected: list[torch.Tensor] | None = None
    best_n_projected = 0

    for set_idx, (set_l, set_u) in enumerate(zip(bounds_l_sets, bounds_u_sets)):
        distance = 0.0
        n_projected = 0
        projected_for_set: list[torch.Tensor] = []

        for p_idx, (param, lb, ub) in enumerate(zip(actor_params, set_l, set_u)):
            p = param.data
            if tuple(lb.shape) != tuple(p.shape) or tuple(ub.shape) != tuple(p.shape):
                raise ValueError(
                    f"Shape mismatch at set {set_idx}, param {p_idx}: "
                    f"param={tuple(p.shape)}, lower={tuple(lb.shape)}, upper={tuple(ub.shape)}",
                )

            projected = torch.maximum(torch.minimum(p, ub), lb)
            outside = (p < lb) | (p > ub)
            n_projected += int(outside.sum().item())
            delta = projected - p
            if norm == "l2":
                # Compare using squared L2 distance (equivalent ordering to L2).
                distance += float(torch.sum(delta * delta).item())
            elif norm == "l1":
                distance += float(torch.sum(torch.abs(delta)).item())
            else:  # norm == "linf"
                distance = max(distance, float(torch.max(torch.abs(delta)).item()))
            projected_for_set.append(projected)

            # Optional pruning: once this candidate is already worse, no need to keep accumulating.
            if distance > best_distance:
                break

        if distance < best_distance:
            best_distance = distance
            best_set_idx = set_idx
            best_projected = projected_for_set
            best_n_projected = n_projected
            if best_distance == 0.0:
                break

    if best_set_idx is None or best_projected is None:
        raise RuntimeError("Failed to select a nearest Rashomon set during PGD projection.")

    # Diagnostics for the selected box (computed before overwriting params).
    sel_l = bounds_l_sets[best_set_idx]
    sel_u = bounds_u_sets[best_set_idx]
    squared_displacement = 0.0
    displacement_linf = 0.0
    n_boundary = 0
    for param, projected, lb, ub in zip(actor_params, best_projected, sel_l, sel_u):
        delta = projected - param.data
        squared_displacement += float(torch.sum(delta * delta).item())
        if delta.numel():
            displacement_linf = max(displacement_linf, float(torch.max(torch.abs(delta)).item()))
        on_face = (projected == lb) | (projected == ub)
        n_boundary += int(on_face.sum().item())

    for param, projected in zip(actor_params, best_projected):
        param.data.copy_(projected)

    return ProjectionResult(
        n_projected=int(best_n_projected),
        n_boundary=int(n_boundary),
        selected_set_index=int(best_set_idx),
        displacement_l2=float(squared_displacement ** 0.5),
        displacement_linf=float(displacement_linf),
    )
