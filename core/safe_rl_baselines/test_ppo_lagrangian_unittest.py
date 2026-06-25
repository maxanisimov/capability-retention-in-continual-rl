"""Tests for the PyTorch PPO-Lagrangian baseline."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from safe_rl_baselines.ppo_lagrangian import PPOLagrangian, _gae


# Cost = 1 whenever the agent takes action 0 ("push left").
def _left_cost(obs, action, *rest) -> float:
    return 1.0 if int(action) == 0 else 0.0


class HelperTests(unittest.TestCase):
    def test_gae_matches_hand_computation(self) -> None:
        rewards = np.array([1.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.5], dtype=np.float32)
        dones = np.array([0.0, 1.0], dtype=np.float32)  # episode ends at t=1
        adv, ret = _gae(rewards, values, dones, last_value=2.0, gamma=1.0, gae_lambda=1.0)
        # t=1 (terminal): delta = 1 + 0 - 0.5 = 0.5 ; adv1 = 0.5
        # t=0: delta = 1 + 1*0.5 - 0.5 = 1.0 ; adv0 = 1.0 + 1*1*0.5 = 1.5
        self.assertTrue(np.allclose(adv, [1.5, 0.5]))
        self.assertTrue(np.allclose(ret, adv + values))

    def test_lambda_update_dynamics(self) -> None:
        m = PPOLagrangian.__new__(PPOLagrangian)  # bypass __init__ for a pure-logic check
        m.lambda_lr = 0.1
        m.cost_limit = 5.0
        m.lagrangian_upper_bound = None
        m.lagrangian_multiplier = 0.0
        m._update_lagrange_multiplier(10.0)               # cost > limit -> grows
        self.assertAlmostEqual(m.lagrangian_multiplier, 0.5)
        m._update_lagrange_multiplier(0.0)                # cost < limit -> shrinks, clamped >= 0
        self.assertAlmostEqual(m.lagrangian_multiplier, 0.0)
        m.lagrangian_upper_bound = 0.05
        m.lagrangian_multiplier = 0.0
        m._update_lagrange_multiplier(10.0)               # capped by upper bound
        self.assertAlmostEqual(m.lagrangian_multiplier, 0.05)


class PPOLagrangianTests(unittest.TestCase):
    def _make(self, cost_limit: float) -> PPOLagrangian:
        return PPOLagrangian(
            gym.make("CartPole-v1"), cost_fn=_left_cost, cost_limit=cost_limit,
            n_steps=1024, batch_size=128, n_epochs=4, lambda_lr=0.1, seed=0,
            device="cpu", net_arch=(64, 64),
        )

    @staticmethod
    def _left_freq(model: PPOLagrangian) -> float:
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

    def test_predict_shape(self) -> None:
        model = self._make(cost_limit=0.0)
        action, state = model.predict(np.zeros(4, dtype=np.float32), deterministic=True)
        self.assertIsNone(state)
        self.assertIn(int(np.asarray(action)), (0, 1))

    def test_constraint_shapes_policy(self) -> None:
        tight = self._make(cost_limit=0.0)      # forbid the costly action
        tight.learn(total_timesteps=6144)
        loose = self._make(cost_limit=1e3)      # effectively unconstrained
        loose.learn(total_timesteps=6144)

        self.assertGreater(tight.lagrangian_multiplier, loose.lagrangian_multiplier)
        self.assertEqual(loose.lagrangian_multiplier, 0.0)
        self.assertLess(self._left_freq(tight), self._left_freq(loose))
        self.assertIn("mean_episode_cost", tight.last_stats)


class CostFromInfoTests(unittest.TestCase):
    def test_reads_cost_from_info(self) -> None:
        class CostInInfo(gym.Wrapper):
            def step(self, action):
                obs, r, term, trunc, info = self.env.step(action)
                info = {**info, "cost": 1.0 if int(action) == 0 else 0.0}
                return obs, r, term, trunc, info

        model = PPOLagrangian(
            CostInInfo(gym.make("CartPole-v1")), cost_limit=0.0, n_steps=512, batch_size=128,
            n_epochs=2, lambda_lr=0.1, seed=0, device="cpu",
        )
        model.learn(total_timesteps=1536)            # no cost_fn -> cost comes from info["cost"]
        self.assertGreater(model.lagrangian_multiplier, 0.0)


class BoxActionTests(unittest.TestCase):
    def test_runs_on_continuous_actions(self) -> None:
        model = PPOLagrangian(
            gym.make("Pendulum-v1"), cost_limit=10.0, n_steps=256, batch_size=64,
            n_epochs=2, seed=0, device="cpu",
        )
        model.learn(total_timesteps=512)
        action, _ = model.predict(np.zeros(3, dtype=np.float32), deterministic=True)
        self.assertEqual(np.asarray(action).shape, (1,))


if __name__ == "__main__":
    unittest.main()
