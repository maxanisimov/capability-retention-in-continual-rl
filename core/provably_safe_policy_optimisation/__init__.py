"""Provably-safe policy optimisation.

Projected-gradient-descent training for Stable-Baselines3 agents: parameters are
projected, after every optimizer step, onto a union of per-parameter boxes (a
"Rashomon set") so the policy provably stays within a certified region.

Public API
----------
* :class:`ProjectedAdam` -- Adam variant that projects (a subset of) its
  parameters each step.
* :class:`ProjectedDQN` / :class:`ProjectedPPO` -- SB3 DQN/PPO subclasses that
  wire ``ProjectedAdam`` in and attach caller-supplied bounds.
* :func:`projection_target_parameter_names` -- resolve the ordered parameter
  names a ``ProjectedPPO`` would project (to align bounds).
* :func:`project_to_interval_union` / :func:`validate_and_prepare_param_interval_bounds`
  -- the low-level projection primitives, also usable standalone.
* policy-introspection helpers for SB3 ActorCritic policies.
"""

from __future__ import annotations

# Pure-torch primitives (no Stable-Baselines3 dependency).
from provably_safe_policy_optimisation.projection import (
    ActorParamBounds,
    project_to_interval_union,
    validate_and_prepare_param_interval_bounds,
)
from provably_safe_policy_optimisation.projected_optimizers import ProjectedAdam
from provably_safe_policy_optimisation.policy_introspection import (
    extract_feature_actor_parameters_and_network,
    resolve_feature_actor_names_for_policy,
    resolve_policy,
)

# SB3-dependent training classes (optional: only if stable-baselines3 installed).
try:
    from provably_safe_policy_optimisation.projected_dqn import ProjectedDQN
    from provably_safe_policy_optimisation.projected_ppo import (
        ProjectedPPO,
        projection_target_parameter_names,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ProjectedDQN = None
    ProjectedPPO = None
    projection_target_parameter_names = None

__all__ = [
    "ActorParamBounds",
    "ProjectedAdam",
    "ProjectedDQN",
    "ProjectedPPO",
    "extract_feature_actor_parameters_and_network",
    "project_to_interval_union",
    "projection_target_parameter_names",
    "resolve_feature_actor_names_for_policy",
    "resolve_policy",
    "validate_and_prepare_param_interval_bounds",
]
