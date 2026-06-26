"""Tests for the PyTorch CPO baseline."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch as th

from safe_rl_baselines import CPO


def _left_cost(obs, action, *rest) -> float:
    return 1.0 if int(action) == 0 else 0.0


def _make(cost_limit: float, **extra) -> CPO:
    return CPO(
        gym.make("CartPole-v1"), cost_fn=_left_cost, cost_limit=cost_limit,
        n_steps=2000, target_kl=0.02, cg_iters=10, n_critic_updates=40,
        seed=0, device="cpu", net_arch=(64, 64), **extra,
    )


class BuildingBlockTests(unittest.TestCase):
    def test_conjugate_gradient_solves_spd(self) -> None:
        A = th.tensor([[4.0, 1.0], [1.0, 3.0]])
        b = th.tensor([1.0, 2.0])
        x = CPO._conjugate_gradient(lambda v: A @ v, b, iters=20)
        self.assertTrue(th.allclose(A @ x, b, atol=1e-5))

    def test_flat_params_round_trip(self) -> None:
        m = _make(5.0)
        flat = m._flat_params().clone()
        m._set_flat_params(flat + 0.123)
        self.assertTrue(th.allclose(m._flat_params(), flat + 0.123))
        m._set_flat_params(flat)
        self.assertTrue(th.allclose(m._flat_params(), flat))

    def test_kl_zero_at_identity_and_fvp_psd(self) -> None:
        m = _make(5.0)
        obs = th.as_tensor(np.random.randn(16, 4).astype(np.float32))
        old = m._old_outputs(obs)
        self.assertAlmostEqual(float(m._mean_kl(obs, old).detach()), 0.0, places=6)
        v = th.randn_like(m._flat_params())
        hv = m._fisher_vector_product(obs, old, v)
        self.assertTrue(th.isfinite(hv).all())
        self.assertGreaterEqual(float(th.dot(v, hv)), 0.0)   # PSD (with damping)

    def test_determine_case(self) -> None:
        m = _make(5.0)
        m.target_kl = 0.01
        # no cost gradient + feasible -> case 4
        self.assertEqual(m._determine_case(0.0, c=-1.0, q=1.0, r=0.0, s=0.0)[0], 4)
        # c<0, B<0 (entire trust region feasible) -> case 3
        self.assertEqual(m._determine_case(1.0, c=-1.0, q=1.0, r=0.0, s=10.0)[0], 3)
        # c<0, B>=0 -> case 2
        self.assertEqual(m._determine_case(1.0, c=-0.01, q=1.0, r=0.0, s=10.0)[0], 2)
        # c>=0, B>=0 -> case 1
        self.assertEqual(m._determine_case(1.0, c=0.01, q=1.0, r=0.0, s=10.0)[0], 1)
        # c>=0, B<0 (infeasible) -> case 0 (recovery)
        self.assertEqual(m._determine_case(1.0, c=10.0, q=1.0, r=0.0, s=10.0)[0], 0)


class CPOIntegrationTests(unittest.TestCase):
    @staticmethod
    def _left_freq(model: CPO) -> float:
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

    def test_constraint_shapes_policy_and_respects_kl(self) -> None:
        tight = _make(cost_limit=0.0)
        tight.learn(total_timesteps=20000)
        loose = _make(cost_limit=1e3)
        loose.learn(total_timesteps=20000)

        self.assertLess(self._left_freq(tight), self._left_freq(loose))
        self.assertLess(self._left_freq(tight), 0.25)
        # An accepted update stays inside the KL trust region.
        if tight.last_stats["accepted"] == 1.0:
            self.assertLess(tight.last_stats["kl"], tight.target_kl)
        for key in ("optim_case", "lambda_star", "nu_star", "constraint_value_c"):
            self.assertIn(key, tight.last_stats)

    def test_predict_shape(self) -> None:
        model = _make(cost_limit=5.0)
        action, state = model.predict(np.zeros(4, dtype=np.float32), deterministic=True)
        self.assertIsNone(state)
        self.assertIn(int(np.asarray(action)), (0, 1))


if __name__ == "__main__":
    unittest.main()
