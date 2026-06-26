"""PID-Lagrangian PPO: a PyTorch safe-RL baseline.

Implements the PID Lagrangian method of Stooke, Achiam & Abbeel (2020),
"Responsive Safety in Reinforcement Learning by PID Lagrangian Methods"
(arXiv:2007.03964). It is identical to :class:`PPOLagrangian` except for how the
Lagrange multiplier is updated: instead of the *integral-only* (naive) update
``lambda <- max(0, lambda + lambda_lr * (J_C - d))`` -- which tends to oscillate
and overshoot the cost limit -- the multiplier is the output of a **PID
controller** on the constraint violation ``delta = J_C - d``:

    P = K_P * delta                         (proportional, fast response)
    I = max(0, I + K_I * delta)             (integral, the naive term)
    D = K_D * max(0, d/dt J_C)              (derivative, damps overshoot)
    lambda = max(0, P + I + D)

The proportional term may be EMA-smoothed and the derivative is taken on an
EMA-smoothed cost signal (and rectified, so it only reacts to *rising* cost), as
is standard for a noisy cost estimate. The naive Lagrangian is the special case
``K_P = K_D = 0, K_I = lambda_lr``.

Everything else -- cost critic, separate reward/cost GAE, combined advantage
``(A_reward - lambda * A_cost)/(1+lambda)``, and the PPO training loop -- is
inherited unchanged from :class:`PPOLagrangian`.
"""

from __future__ import annotations

from typing import Any

from safe_rl_baselines.ppo_lagrangian import PPOLagrangian


class PPOPIDLagrangian(PPOLagrangian):
    """PPO-Lagrangian with a PID controller for the Lagrange multiplier."""

    def __init__(
        self,
        env: Any,
        *,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.01,
        pid_p_ema_alpha: float = 0.0,
        pid_d_ema_alpha: float = 0.95,
        penalty_max: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env, **kwargs)
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        # EMA smoothing factors: alpha is the weight on the *previous* value, so
        # alpha=0 means no smoothing (use the raw signal).
        self.pid_p_ema_alpha = pid_p_ema_alpha
        self.pid_d_ema_alpha = pid_d_ema_alpha
        self.penalty_max = penalty_max if penalty_max is not None else self.lagrangian_upper_bound

        # PID state. The integral is seeded with the initial multiplier so that,
        # in the naive special case, this reproduces PPOLagrangian exactly.
        self._pid_i = float(self.lagrangian_multiplier)
        self._delta_p = 0.0
        self._cost_d = 0.0
        self._prev_cost_d = 0.0
        self.pid_terms: dict[str, float] = {"P": 0.0, "I": self._pid_i, "D": 0.0}

    def _update_lagrange_multiplier(self, mean_ep_cost: float) -> None:
        delta = mean_ep_cost - self.cost_limit

        # Integral term (rectified to stay non-negative).
        self._pid_i = max(0.0, self._pid_i + self.pid_ki * delta)

        # Proportional term (optionally EMA-smoothed violation).
        a_p = self.pid_p_ema_alpha
        self._delta_p = a_p * self._delta_p + (1.0 - a_p) * delta
        p_term = self.pid_kp * self._delta_p

        # Derivative term on the EMA-smoothed cost, rectified so it only responds
        # to *rising* cost (damping overshoot without loosening on improvement).
        a_d = self.pid_d_ema_alpha
        self._cost_d = a_d * self._cost_d + (1.0 - a_d) * mean_ep_cost
        d_term = self.pid_kd * max(0.0, self._cost_d - self._prev_cost_d)
        self._prev_cost_d = self._cost_d

        multiplier = max(0.0, p_term + self._pid_i + d_term)
        if self.penalty_max is not None:
            multiplier = min(multiplier, self.penalty_max)
        self.lagrangian_multiplier = multiplier
        self.pid_terms = {"P": float(p_term), "I": float(self._pid_i), "D": float(d_term)}

    def train(self) -> None:
        super().train()
        # The inherited train() rebuilds last_stats; surface the PID terms here.
        self.last_stats["lambda_P"] = self.pid_terms["P"]
        self.last_stats["lambda_I"] = self.pid_terms["I"]
        self.last_stats["lambda_D"] = self.pid_terms["D"]
