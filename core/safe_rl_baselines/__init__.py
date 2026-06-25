"""Safe-RL baselines (constrained MDP) implemented in PyTorch.

Currently provides :class:`PPOLagrangian` -- a PyTorch port of MASA's JAX ``PPOLag``.
"""

from __future__ import annotations

from safe_rl_baselines.ppo_lagrangian import PPOLagrangian

__all__ = ["PPOLagrangian"]
