"""Stable-Baselines3 DQN with projected gradient descent onto parameter bounds.

:class:`ProjectedDQN` is a thin subclass of SB3's :class:`~stable_baselines3.DQN`
that, after every gradient update, projects the online Q-network parameters onto
a union of per-parameter boxes (a "Rashomon set"). The projection reuses the
geometry of :mod:`provably_safe_policy_optimisation.projection` and is
implemented as a custom optimizer,
:class:`~provably_safe_policy_optimisation.projected_optimizers.ProjectedAdam`.

The bounds are **caller-supplied** and must be aligned, in count, shape, and
order, with ``model.policy.q_net.parameters()``. Both forms are accepted:

* single box: ``list[Tensor]`` (one lower/upper tensor per parameter), or
* union of boxes: ``list[list[Tensor]]`` (interval-major or parameter-major).

Usage
-----
>>> from provably_safe_policy_optimisation import ProjectedDQN
>>> model = ProjectedDQN(
...     "MlpPolicy", env,
...     param_bounds_l=lower, param_bounds_u=upper,  # aligned to q_net.parameters()
...     projection_distance_norm="l2",
...     policy_kwargs={"net_arch": [64, 64]},
... )
>>> model.learn(total_timesteps=100_000)

Escape hatch (no subclass): pass ``ProjectedAdam`` directly and attach bounds
after construction::

    model = DQN("MlpPolicy", env, policy_kwargs=dict(
        optimizer_class=ProjectedAdam,
        optimizer_kwargs=dict(distance_norm="l2"),
    ))
    model.policy.optimizer.set_bounds(lower, upper)

Save/Load notes
---------------
Projection bounds are *not* persisted in the policy state_dict. After
``ProjectedDQN.load(...)`` (or ``DQN.load`` when ``ProjectedAdam`` is importable),
re-attach bounds via ``model.policy.optimizer.set_bounds(lower, upper)`` to
resume constrained training. Only the online ``q_net`` is projected; the target
network is a periodic copy of the (already projected) ``q_net``, so it needs no
special handling.
"""

from __future__ import annotations

from typing import Any

from stable_baselines3 import DQN

from provably_safe_policy_optimisation._projection_logging import record_projection_window
from provably_safe_policy_optimisation.projected_optimizers import ProjectedAdam
from provably_safe_policy_optimisation.projection import ActorParamBounds


class ProjectedDQN(DQN):
    """DQN that projects ``q_net`` parameters onto a union of boxes each step."""

    def __init__(
        self,
        *args: Any,
        param_bounds_l: ActorParamBounds | None = None,
        param_bounds_u: ActorParamBounds | None = None,
        projection_distance_norm: str = "l2",
        policy_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if (param_bounds_l is None) != (param_bounds_u is None):
            raise ValueError(
                "param_bounds_l and param_bounds_u must both be provided or both be None.",
            )

        # Bounds must be available before super().__init__ runs, because it calls
        # self._setup_model() (where we attach them to the built optimizer).
        self._param_bounds_l = param_bounds_l
        self._param_bounds_u = param_bounds_u

        # Force the projected optimizer. We create it bounds-free (a no-op) and
        # attach bounds after the policy/optimizer exist, sidestepping the fact
        # that parameter tensors do not exist at optimizer-construction time.
        policy_kwargs = dict(policy_kwargs or {})
        existing_optimizer_class = policy_kwargs.get("optimizer_class")
        if existing_optimizer_class not in (None, ProjectedAdam):
            raise ValueError(
                "ProjectedDQN manages its own optimizer_class (ProjectedAdam); "
                f"got optimizer_class={existing_optimizer_class!r} in policy_kwargs.",
            )
        policy_kwargs["optimizer_class"] = ProjectedAdam
        optimizer_kwargs = dict(policy_kwargs.get("optimizer_kwargs") or {})
        optimizer_kwargs["distance_norm"] = projection_distance_norm
        policy_kwargs["optimizer_kwargs"] = optimizer_kwargs

        super().__init__(*args, policy_kwargs=policy_kwargs, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self._param_bounds_l is not None and self._param_bounds_u is not None:
            # set_bounds validates count/shape/order against q_net.parameters()
            # and raises a clear error on misalignment.
            self.policy.optimizer.set_bounds(self._param_bounds_l, self._param_bounds_u)

    def projection_diagnostics(self) -> dict[str, Any]:
        """Cumulative projection diagnostics (see ``ProjectedAdam``)."""
        return self.policy.optimizer.projection_diagnostics()

    def train(self, *args: Any, **kwargs: Any) -> Any:
        result = super().train(*args, **kwargs)
        record_projection_window(self)
        return result
