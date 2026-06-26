"""Safe-RL baselines (constrained MDP) implemented in PyTorch.

- :class:`PPOLagrangian` -- a PyTorch port of MASA's JAX ``PPOLag`` (naive
  integral Lagrange dual update).
- :class:`PPOPIDLagrangian` -- PID Lagrangian (Stooke et al. 2020).
"""

from __future__ import annotations

from safe_rl_baselines.ppo_lagrangian import PPOLagrangian
from safe_rl_baselines.ppo_pid_lagrangian import PPOPIDLagrangian

__all__ = ["PPOLagrangian", "PPOPIDLagrangian"]
