"""Tests for the PyTorch PID-Lagrangian PPO baseline."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from safe_rl_baselines.ppo_lagrangian import PPOLagrangian
from safe_rl_baselines.ppo_pid_lagrangian import PPOPIDLagrangian


def _left_cost(obs, action, *rest) -> float:
    return 1.0 if int(action) == 0 else 0.0


class _Controller(PPOPIDLagrangian):
    """Construct only the PID controller state, bypassing env/network setup."""

    def __init__(self, *, pid_kp, pid_ki, pid_kd, cost_limit, init=0.0,
                 pid_d_ema_alpha=0.0, pid_p_ema_alpha=0.0, penalty_max=None):
        self.cost_limit = cost_limit
        self.pid_kp, self.pid_ki, self.pid_kd = pid_kp, pid_ki, pid_kd
        self.pid_p_ema_alpha = pid_p_ema_alpha
        self.pid_d_ema_alpha = pid_d_ema_alpha
        self.penalty_max = penalty_max
        self.lagrangian_multiplier = init
        self._pid_i = float(init)
        self._delta_p = 0.0
        self._cost_d = 0.0
        self._prev_cost_d = 0.0
        self.pid_terms = {"P": 0.0, "I": float(init), "D": 0.0}


class _NaiveController(PPOLagrangian):
    def __init__(self, *, lambda_lr, cost_limit, init=0.0, upper=None):
        self.lambda_lr = lambda_lr
        self.cost_limit = cost_limit
        self.lagrangian_upper_bound = upper
        self.lagrangian_multiplier = init


class PIDControllerTests(unittest.TestCase):
    def test_reduces_to_naive_lagrangian(self) -> None:
        # K_P = K_D = 0, K_I = lambda_lr, no derivative smoothing -> integral-only.
        pid = _Controller(pid_kp=0.0, pid_ki=0.05, pid_kd=0.0, cost_limit=5.0)
        naive = _NaiveController(lambda_lr=0.05, cost_limit=5.0)
        for cost in [10.0, 8.0, 2.0, 6.0, 0.0, 12.0]:
            pid._update_lagrange_multiplier(cost)
            naive._update_lagrange_multiplier(cost)
            self.assertAlmostEqual(pid.lagrangian_multiplier, naive.lagrangian_multiplier, places=6)

    def test_integral_is_rectified(self) -> None:
        pid = _Controller(pid_kp=0.0, pid_ki=1.0, pid_kd=0.0, cost_limit=5.0)
        pid._update_lagrange_multiplier(0.0)   # delta = -5 -> integral clamped at 0
        self.assertEqual(pid._pid_i, 0.0)
        self.assertEqual(pid.lagrangian_multiplier, 0.0)

    def test_derivative_only_reacts_to_rising_cost(self) -> None:
        pid = _Controller(pid_kp=0.0, pid_ki=0.0, pid_kd=1.0, cost_limit=0.0)
        pid._update_lagrange_multiplier(1.0)   # cost rising from 0 -> positive D term
        rising = pid.pid_terms["D"]
        self.assertGreater(rising, 0.0)
        pid2 = _Controller(pid_kp=0.0, pid_ki=0.0, pid_kd=1.0, cost_limit=0.0)
        pid2._update_lagrange_multiplier(5.0)
        pid2._update_lagrange_multiplier(1.0)  # cost falling -> rectified D term is 0
        self.assertEqual(pid2.pid_terms["D"], 0.0)

    def test_penalty_max_clamps(self) -> None:
        pid = _Controller(pid_kp=1.0, pid_ki=1.0, pid_kd=0.0, cost_limit=0.0, penalty_max=0.5)
        pid._update_lagrange_multiplier(100.0)
        self.assertLessEqual(pid.lagrangian_multiplier, 0.5)

    def test_persistent_violation_grows_multiplier(self) -> None:
        pid = _Controller(pid_kp=0.1, pid_ki=0.05, pid_kd=0.01, cost_limit=5.0)
        start = pid.lagrangian_multiplier
        for _ in range(10):
            pid._update_lagrange_multiplier(20.0)  # persistently over the limit
        self.assertGreater(pid.lagrangian_multiplier, start)


class PPOPIDLagrangianIntegrationTests(unittest.TestCase):
    def _make(self, cost_limit: float) -> PPOPIDLagrangian:
        return PPOPIDLagrangian(
            gym.make("CartPole-v1"), cost_fn=_left_cost, cost_limit=cost_limit,
            n_steps=1024, batch_size=128, n_epochs=4, seed=0, device="cpu",
            net_arch=(64, 64), pid_kp=0.1, pid_ki=0.05, pid_kd=0.01,
        )

    @staticmethod
    def _left_freq(model: PPOPIDLagrangian) -> float:
        env = gym.make("CartPole-v1")
        left = total = 0
        for ep in range(10):
            obs, _ = env.reset(seed=100 + ep)
            done = False
            while not done:
                a = int(np.asarray(model.predict(obs, deterministic=True)[0]))
                left += int(a == 0); total += 1
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
        return left / total

    def test_constraint_shapes_policy_and_reports_pid_terms(self) -> None:
        tight = self._make(cost_limit=0.0)
        tight.learn(total_timesteps=6144)
        loose = self._make(cost_limit=1e3)
        loose.learn(total_timesteps=6144)

        self.assertGreater(tight.lagrangian_multiplier, loose.lagrangian_multiplier)
        self.assertEqual(loose.lagrangian_multiplier, 0.0)
        self.assertLess(self._left_freq(tight), self._left_freq(loose))
        for key in ("lambda_P", "lambda_I", "lambda_D"):
            self.assertIn(key, tight.last_stats)


if __name__ == "__main__":
    unittest.main()
