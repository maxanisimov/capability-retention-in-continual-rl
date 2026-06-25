"""Projected-gradient-descent optimizers for SB3-style training.

This module provides :class:`ProjectedAdam`, a drop-in replacement for
``torch.optim.Adam`` that, after every optimizer step, projects the optimized
parameters onto a union of per-parameter boxes (a "Rashomon set"). This is the
same projection used by the custom PPO trainer in
:mod:`experiments.utils.ppo_utils`; here it is exposed as an optimizer so it can
be plugged into Stable-Baselines3 algorithms (e.g. DQN) via ``optimizer_class``.

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
  is delegated to the existing, battle-tested helpers in ``ppo_utils`` so the
  geometry stays identical to the PPO Rashomon path.
"""

from __future__ import annotations

from typing import Any

import torch

from experiments.utils.ppo_utils import (
    ActorParamBounds,
    _project_actor_to_interval_union,
    _validate_and_prepare_param_interval_bounds,
)


class ProjectedAdam(torch.optim.Adam):
    """Adam optimizer that projects parameters onto a union of boxes each step.

    Parameters
    ----------
    params:
        Iterable of parameters (or a single param group of plain tensors), as
        passed by SB3. Param-group dicts are not supported (SB3 DQN passes
        ``q_net.parameters()``).
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
        # Lightweight telemetry, mirroring ClampedPPO's counters.
        self._step_calls = 0
        self._projected_elements = 0

    @property
    def has_bounds(self) -> bool:
        """Whether projection is currently active."""
        return self._bounds_l_sets is not None

    def set_bounds(
        self,
        bounds_l: ActorParamBounds,
        bounds_u: ActorParamBounds,
        params: list[torch.nn.Parameter] | None = None,
    ) -> None:
        """Attach projection bounds, aligned to a parameter order.

        Accepts either the single-box form (``list[Tensor]`` with one lower/upper
        tensor per parameter) or the union-of-boxes form (``list[list[Tensor]]``,
        interval-major or parameter-major). Bounds are validated against the
        parameter shapes/order and moved onto the parameters' device.

        Parameters
        ----------
        params:
            Ordered subset of this optimizer's parameters to project, and the
            order the bounds are aligned to. Defaults to *all* of the optimizer's
            parameters. Useful when one optimizer covers more than the
            constrained sub-network (e.g. SB3 PPO's single actor+critic
            optimizer, where only the actor should be projected). Every entry
            must be one of the optimizer's own parameters.
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
        l_sets, u_sets = _validate_and_prepare_param_interval_bounds(
            actor_params=target,
            actor_param_bounds_l=bounds_l,
            actor_param_bounds_u=bounds_u,
            device=device,
        )
        self._projection_params = target
        self._bounds_l_sets = l_sets
        self._bounds_u_sets = u_sets

    def clear_bounds(self) -> None:
        """Disable projection (revert to plain Adam behaviour)."""
        self._projection_params = self._projected_params
        self._bounds_l_sets = None
        self._bounds_u_sets = None

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = super().step(closure)
        if self._bounds_l_sets is not None and self._bounds_u_sets is not None:
            n_projected = _project_actor_to_interval_union(
                self._projection_params,
                self._bounds_l_sets,
                self._bounds_u_sets,
                distance_norm=self._distance_norm,
            )
            self._step_calls += 1
            self._projected_elements += int(n_projected)
        return loss
