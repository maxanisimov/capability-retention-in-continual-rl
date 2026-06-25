"""Projected-gradient-descent optimizers for SB3-style training.

This module provides :class:`ProjectedAdam`, a drop-in replacement for
``torch.optim.Adam`` that, after every optimizer step, projects (a subset of)
the optimized parameters onto a union of per-parameter boxes (a "Rashomon set").
This is the same projection used by the custom PPO trainer in
``experiments.utils.ppo_utils``; here it is exposed as an optimizer so it can be
plugged into Stable-Baselines3 algorithms (e.g. DQN, PPO) via ``optimizer_class``.

Design notes
------------
* SB3 instantiates the optimizer as ``optimizer_class(net.parameters(), lr=...,
  **optimizer_kwargs)``. The parameter tensors therefore do not exist until the
  policy is built, so bounds cannot be keyed by parameter object up front.
  Instead, bounds are **param-order-aligned** lists and are attached *after*
  construction via :meth:`ProjectedAdam.set_bounds`.
* Until bounds are set the optimizer behaves exactly like ``torch.optim.Adam``
  (projection is a no-op), so it is always safe to pass as ``optimizer_class``.
* The heavy lifting (bound validation/normalization and nearest-box projection)
  is delegated to :mod:`provably_safe_policy_optimisation.projection` so the
  geometry stays identical to the PPO Rashomon path.
"""

from __future__ import annotations

from typing import Any

import torch

from provably_safe_policy_optimisation.projection import (
    ActorParamBounds,
    ProjectionResult,
    project_to_interval_union,
    validate_and_prepare_param_interval_bounds,
)


class ProjectedAdam(torch.optim.Adam):
    """Adam optimizer that projects parameters onto a union of boxes each step.

    Parameters
    ----------
    params:
        Iterable of parameters (or a single param group of plain tensors), as
        passed by SB3. Param-group dicts are not supported (SB3 passes
        ``net.parameters()``).
    distance_norm:
        Norm used to pick the nearest box when more than one box is supplied.
        One of ``"l2"``, ``"l1"``, ``"linf"``.
    **adam_kwargs:
        Forwarded to ``torch.optim.Adam`` (this is where SB3's ``lr`` arrives).

    Notes
    -----
    Call :meth:`set_bounds` (typically right after the owning policy is built)
    to enable projection. Before that, ``step`` is identical to Adam's.
    """

    def __init__(
        self,
        params: Any,
        *,
        distance_norm: str = "l2",
        **adam_kwargs: Any,
    ) -> None:
        # Materialize once: SB3 passes a generator that must not be consumed by
        # ``super().__init__`` before we can keep an ordered reference for
        # projection.
        self._projected_params: list[torch.nn.Parameter] = list(params)
        super().__init__(self._projected_params, **adam_kwargs)

        self._distance_norm = str(distance_norm)
        # Ordered subset of parameters that projection targets (defaults to all
        # of this optimizer's parameters; PPO uses a subset, see set_bounds).
        self._projection_params: list[torch.nn.Parameter] = self._projected_params
        self._bounds_l_sets: list[list[torch.Tensor]] | None = None
        self._bounds_u_sets: list[list[torch.Tensor]] | None = None
        # Diagnostics (cumulative over all bounded steps; see projection_diagnostics).
        self._step_calls = 0                  # optimizer steps taken while bounds active
        self._projection_active_steps = 0     # steps where >=1 element was clamped
        self._projected_elements = 0          # cumulative out-of-bounds entries clamped
        self._boundary_elements = 0           # cumulative entries on a box face after projection
        self._displacement_l2_sum = 0.0       # sum of per-step L2 projection magnitudes
        self._displacement_linf_sum = 0.0     # sum of per-step L-inf projection magnitudes
        self._max_displacement_l2 = 0.0       # running max per-step L2 magnitude
        self._max_displacement_linf = 0.0     # running max per-step L-inf magnitude
        self._selected_box_counts: dict[int, int] = {}  # histogram of selected box index
        # Outcome of the initial feasibility projection done by set_bounds (if any);
        # kept separate from the per-step diagnostics (it is not a gradient update).
        self._init_projection: ProjectionResult | None = None

    @property
    def has_bounds(self) -> bool:
        """Whether projection is currently active."""
        return self._bounds_l_sets is not None

    def set_bounds(
        self,
        bounds_l: ActorParamBounds,
        bounds_u: ActorParamBounds,
        params: list[torch.nn.Parameter] | None = None,
        project_on_set: bool = True,
    ) -> None:
        """Attach projection bounds, aligned to a parameter order.

        Accepts either the single-box form (``list[Tensor]`` with one lower/upper
        tensor per parameter) or the union-of-boxes form (``list[list[Tensor]]``,
        interval-major or parameter-major). Bounds are validated against the
        parameter shapes/order and moved onto the parameters' device and dtype.

        Parameters
        ----------
        params:
            Ordered subset of this optimizer's parameters to project, and the
            order the bounds are aligned to. Defaults to *all* of the optimizer's
            parameters. Useful when one optimizer covers more than the
            constrained sub-network (e.g. SB3 PPO's single actor+critic
            optimizer, where only the actor should be projected). Every entry
            must be one of the optimizer's own parameters.
        project_on_set:
            If ``True`` (default), immediately project the current parameters into
            the bounds so the **starting point is feasible** (recorded in
            ``self._init_projection``; not counted as an optimizer step). Set
            ``False`` to leave parameters untouched until the first ``step``.
        """
        target = self._projected_params if params is None else list(params)
        if not target:
            raise ValueError("set_bounds requires at least one parameter to project.")
        owned = {id(p) for p in self._projected_params}
        if any(id(p) not in owned for p in target):
            raise ValueError(
                "Every parameter passed to set_bounds(params=...) must belong to "
                "this optimizer (so its updates are actually projected).",
            )

        device = target[0].device
        l_sets, u_sets = validate_and_prepare_param_interval_bounds(
            actor_params=target,
            actor_param_bounds_l=bounds_l,
            actor_param_bounds_u=bounds_u,
            device=device,
        )
        self._projection_params = target
        self._bounds_l_sets = l_sets
        self._bounds_u_sets = u_sets
        if project_on_set:
            self._init_projection = self.project_now()

    def clear_bounds(self) -> None:
        """Disable projection (revert to plain Adam behaviour)."""
        self._projection_params = self._projected_params
        self._bounds_l_sets = None
        self._bounds_u_sets = None

    @torch.no_grad()
    def project_now(self) -> ProjectionResult:
        """Project the current parameters into the bounds, on demand.

        Use after manually changing weights (e.g. warm-starting after the model
        was built, or re-attaching bounds following ``load``) to re-establish
        feasibility. Does not affect the per-step diagnostics counters.
        """
        if self._bounds_l_sets is None or self._bounds_u_sets is None:
            raise RuntimeError("project_now() called before set_bounds().")
        return project_to_interval_union(
            self._projection_params,
            self._bounds_l_sets,
            self._bounds_u_sets,
            distance_norm=self._distance_norm,
        )

    @torch.no_grad()
    def max_violation(self) -> float:
        """Largest bound violation of the current parameters (0.0 == feasible).

        Union-aware: returns the *minimum* over boxes of that box's worst
        single-coordinate violation, since the parameters need only lie inside
        one box of the union. ``0.0`` when no bounds are attached.
        """
        if self._bounds_l_sets is None or self._bounds_u_sets is None:
            return 0.0
        best = float("inf")
        for set_l, set_u in zip(self._bounds_l_sets, self._bounds_u_sets):
            worst = 0.0
            for param, lb, ub in zip(self._projection_params, set_l, set_u):
                p = param.data
                over = torch.clamp(p - ub, min=0.0)
                under = torch.clamp(lb - p, min=0.0)
                if over.numel():
                    worst = max(worst, float(torch.max(over).item()))
                    worst = max(worst, float(torch.max(under).item()))
            best = min(best, worst)
        return float(best)

    def is_within_bounds(self, atol: float = 0.0) -> bool:
        """Whether the current parameters satisfy the bounds (within ``atol``)."""
        return self.max_violation() <= atol

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = super().step(closure)
        if self._bounds_l_sets is not None and self._bounds_u_sets is not None:
            result = project_to_interval_union(
                self._projection_params,
                self._bounds_l_sets,
                self._bounds_u_sets,
                distance_norm=self._distance_norm,
            )
            self._record(result)
        return loss

    def _record(self, result: ProjectionResult) -> None:
        """Fold a single-step :class:`ProjectionResult` into cumulative counters."""
        self._step_calls += 1
        if result.n_projected > 0:
            self._projection_active_steps += 1
        self._projected_elements += int(result.n_projected)
        self._boundary_elements += int(result.n_boundary)
        self._displacement_l2_sum += float(result.displacement_l2)
        self._displacement_linf_sum += float(result.displacement_linf)
        self._max_displacement_l2 = max(self._max_displacement_l2, float(result.displacement_l2))
        self._max_displacement_linf = max(self._max_displacement_linf, float(result.displacement_linf))
        idx = int(result.selected_set_index)
        self._selected_box_counts[idx] = self._selected_box_counts.get(idx, 0) + 1

    @property
    def constrained_element_count(self) -> int:
        """Total number of scalar entries currently under projection."""
        if self._bounds_l_sets is None:
            return 0
        return int(sum(p.numel() for p in self._projection_params))

    def projection_diagnostics(self) -> dict[str, Any]:
        """Return cumulative projection diagnostics.

        Keys
        ----
        bounded_steps:
            Optimizer steps taken while bounds were attached (the denominator).
        projection_active_steps / projection_active_fraction:
            Steps (and their fraction) in which at least one parameter was
            actually outside the bounds and had to be clamped.
        projected_elements_total:
            Cumulative number of scalar entries clamped.
        mean_projected_elements_per_step / mean_projected_elements_per_active_step:
            Average clamped entries per bounded step, and per *active* step.
        constrained_element_count / mean_projected_fraction_per_step:
            Size of the constrained parameter set, and the average fraction of
            it clamped per bounded step.
        mean/max_displacement_l2, mean/max_displacement_linf:
            How far (L2 and L-inf) the projection pushed parameters back into
            bounds, averaged over bounded steps and the running maximum.
        boundary_elements_total / mean_boundary_elements_per_step:
            Entries sitting exactly on a box face after projection (active
            constraints / pinned weights).
        selected_box_counts:
            Histogram {box_index: count} of which box was projected onto (only
            informative for union-of-boxes; always {0: ...} for a single box).
        """
        steps = self._step_calls
        active = self._projection_active_steps
        constrained = self.constrained_element_count
        return {
            "bounded_steps": int(steps),
            "projection_active_steps": int(active),
            "projection_active_fraction": (active / steps) if steps else 0.0,
            "projected_elements_total": int(self._projected_elements),
            "mean_projected_elements_per_step": (self._projected_elements / steps) if steps else 0.0,
            "mean_projected_elements_per_active_step": (self._projected_elements / active) if active else 0.0,
            "constrained_element_count": int(constrained),
            "mean_projected_fraction_per_step": (
                self._projected_elements / (steps * constrained) if steps and constrained else 0.0
            ),
            "mean_displacement_l2": (self._displacement_l2_sum / steps) if steps else 0.0,
            "max_displacement_l2": float(self._max_displacement_l2),
            "mean_displacement_linf": (self._displacement_linf_sum / steps) if steps else 0.0,
            "max_displacement_linf": float(self._max_displacement_linf),
            "boundary_elements_total": int(self._boundary_elements),
            "mean_boundary_elements_per_step": (self._boundary_elements / steps) if steps else 0.0,
            "selected_box_counts": dict(self._selected_box_counts),
        }

    def _raw_diag_counters(self) -> dict[str, float]:
        """Raw cumulative counters used to compute per-window logging deltas."""
        return {
            "bounded_steps": float(self._step_calls),
            "projection_active_steps": float(self._projection_active_steps),
            "projected_elements": float(self._projected_elements),
            "displacement_l2_sum": float(self._displacement_l2_sum),
            "max_displacement_l2": float(self._max_displacement_l2),
        }

    def reset_projection_diagnostics(self) -> None:
        """Zero all cumulative diagnostics (e.g. to measure a fresh window)."""
        self._step_calls = 0
        self._projection_active_steps = 0
        self._projected_elements = 0
        self._boundary_elements = 0
        self._displacement_l2_sum = 0.0
        self._displacement_linf_sum = 0.0
        self._max_displacement_l2 = 0.0
        self._max_displacement_linf = 0.0
        self._selected_box_counts = {}
