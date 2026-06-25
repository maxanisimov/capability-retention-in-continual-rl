"""Stable-Baselines3 PPO with projected gradient descent onto parameter bounds.

:class:`ProjectedPPO` is the PPO analogue of
:class:`experiments.utils.sb3_projected_dqn.ProjectedDQN`. After every gradient
step it projects a chosen subset of the policy parameters onto a union of
per-parameter boxes (a "Rashomon set"), reusing the exact projection geometry of
the PPO Rashomon path (``experiments.utils.ppo_utils``) via the shared
:class:`~experiments.utils.projected_optimizers.ProjectedAdam` optimizer.

Why a subset?
-------------
SB3 PPO builds a *single* optimizer over the whole policy (actor **and** critic).
Retention bounds (e.g. from ``IntervalTrainer``) are computed for the **actor**
only, so by default ``ProjectedPPO`` projects just the actor branch
(``feature_actor`` = actor-side features extractor + ``mlp_extractor.policy_net``
+ ``action_net``). The critic keeps training unconstrained. Set
``projection_target="all"`` to project the entire policy, or pass an explicit
ordered ``projection_param_names`` list.

Aligning bounds
---------------
Caller-supplied bounds must be aligned, in count/shape/order, with the resolved
projection target. Use :func:`projection_target_parameter_names` to obtain that
order for a built model (or build a probe model first), then construct bounds in
the same order. Both forms are accepted: single box (``list[Tensor]``) or union
of boxes (``list[list[Tensor]]``).

Usage
-----
>>> from experiments.utils.sb3_projected_ppo import (
...     ProjectedPPO, projection_target_parameter_names,
... )
>>> probe = ProjectedPPO("MlpPolicy", env)            # discover actor param order
>>> names = projection_target_parameter_names(probe)  # ordered actor param names
>>> # build lower/upper aligned to `names` ...
>>> model = ProjectedPPO("MlpPolicy", env,
...                      param_bounds_l=lower, param_bounds_u=upper,
...                      projection_distance_norm="l2")
>>> model.learn(total_timesteps=100_000)

Escape hatch (no subclass): pass ``ProjectedAdam`` directly and attach bounds to
the actor subset after construction::

    model = PPO("MlpPolicy", env, policy_kwargs=dict(optimizer_class=ProjectedAdam))
    names = projection_target_parameter_names(model)
    name_to_param = dict(model.policy.named_parameters())
    model.policy.optimizer.set_bounds(lower, upper, params=[name_to_param[n] for n in names])

Save/Load notes
---------------
Projection bounds are *not* persisted. After ``ProjectedPPO.load(...)`` (or
``PPO.load`` when ``ProjectedAdam`` is importable), re-attach bounds via
``model.policy.optimizer.set_bounds(...)`` to resume constrained training.
"""

from __future__ import annotations

from typing import Any, Literal

import torch as th
from stable_baselines3 import PPO

from experiments.utils.ppo_utils import ActorParamBounds
from experiments.utils.projected_optimizers import ProjectedAdam
from experiments.utils.sb3_clamped_ppo import (
    _resolve_feature_actor_names_for_policy,
    _resolve_policy,
)

ProjectionTarget = Literal["feature_actor", "all"]


def projection_target_parameter_names(
    model_or_policy: Any,
    *,
    projection_target: ProjectionTarget = "feature_actor",
) -> list[str]:
    """Return the ordered parameter names a ProjectedPPO would project.

    Use this to build bounds aligned to the projection target. ``"feature_actor"``
    returns the actor branch names (sorted, canonical); ``"all"`` returns every
    policy parameter name in ``named_parameters`` order.
    """
    policy = _resolve_policy(model_or_policy)
    if projection_target == "all":
        return [name for name, _ in policy.named_parameters()]
    if projection_target == "feature_actor":
        return _resolve_feature_actor_names_for_policy(policy, dict(policy.named_parameters()))
    raise ValueError(
        f"Unsupported projection_target={projection_target!r}; "
        "expected 'feature_actor' or 'all'.",
    )


class ProjectedPPO(PPO):
    """PPO that projects a subset of policy parameters onto a union of boxes."""

    def __init__(
        self,
        *args: Any,
        param_bounds_l: ActorParamBounds | None = None,
        param_bounds_u: ActorParamBounds | None = None,
        projection_distance_norm: str = "l2",
        projection_target: ProjectionTarget = "feature_actor",
        projection_param_names: list[str] | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if (param_bounds_l is None) != (param_bounds_u is None):
            raise ValueError(
                "param_bounds_l and param_bounds_u must both be provided or both be None.",
            )

        # Bounds/target must exist before super().__init__ runs: it calls
        # self._setup_model(), where we attach bounds to the built optimizer.
        self._param_bounds_l = param_bounds_l
        self._param_bounds_u = param_bounds_u
        self._projection_target: ProjectionTarget = projection_target
        self._projection_param_names = projection_param_names

        policy_kwargs = dict(policy_kwargs or {})
        existing_optimizer_class = policy_kwargs.get("optimizer_class")
        if existing_optimizer_class not in (None, ProjectedAdam):
            raise ValueError(
                "ProjectedPPO manages its own optimizer_class (ProjectedAdam); "
                f"got optimizer_class={existing_optimizer_class!r} in policy_kwargs.",
            )
        policy_kwargs["optimizer_class"] = ProjectedAdam
        optimizer_kwargs = dict(policy_kwargs.get("optimizer_kwargs") or {})
        optimizer_kwargs["distance_norm"] = projection_distance_norm
        policy_kwargs["optimizer_kwargs"] = optimizer_kwargs

        super().__init__(*args, policy_kwargs=policy_kwargs, **kwargs)

    def _resolve_projection_params(self) -> list[th.nn.Parameter]:
        name_to_param = dict(self.policy.named_parameters())
        if self._projection_param_names is not None:
            missing = [n for n in self._projection_param_names if n not in name_to_param]
            if missing:
                raise ValueError(f"projection_param_names not found in policy: {missing}")
            names = self._projection_param_names
        else:
            names = projection_target_parameter_names(
                self.policy, projection_target=self._projection_target
            )
        return [name_to_param[n] for n in names]

    def _setup_model(self) -> None:
        super()._setup_model()
        if self._param_bounds_l is not None and self._param_bounds_u is not None:
            target_params = self._resolve_projection_params()
            # set_bounds validates count/shape/order against target_params and
            # raises a clear error on misalignment.
            self.policy.optimizer.set_bounds(
                self._param_bounds_l, self._param_bounds_u, params=target_params
            )
