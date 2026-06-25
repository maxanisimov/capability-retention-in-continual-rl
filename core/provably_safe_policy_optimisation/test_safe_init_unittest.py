"""Tests for safe initialization (behavioural cloning + IBP-certified greedy-safety)."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from provably_safe_policy_optimisation import ProvablySafeDQN, ProvablySafePPO, RegionShield
from provably_safe_policy_optimisation.safe_init import certify_with_verifier


def _only_action_safe(action: int, n_states: int = 16, n_actions: int = 4) -> np.ndarray:
    mask = np.zeros((n_states, n_actions), dtype=int)
    mask[:, action] = 1
    return mask


def _mc_box_shield() -> RegionShield:
    # MountainCar: position < 0.0 -> only push right (action 2) is safe.
    return RegionShield.from_boxes([([-1.2, -0.07], [0.0, 0.07], [2])], n_actions=3, seed=0)


class CertifyHelperTests(unittest.TestCase):
    def test_certify_agrees_with_brute_force(self) -> None:
        # A net that strongly prefers action 0 over a small box.
        seq = nn.Sequential(nn.Linear(2, 3))
        with th.no_grad():
            seq[0].weight.zero_()
            seq[0].bias.copy_(th.tensor([5.0, 0.0, 0.0]))
        x_l = th.tensor([[-0.1, -0.1]]); x_u = th.tensor([[0.1, 0.1]])
        frac_ok, all_ok = certify_with_verifier(seq, x_l, x_u, th.tensor([[1, 0, 0]]))
        self.assertTrue(all_ok)                       # action 0 certified safe
        frac_bad, all_bad = certify_with_verifier(seq, x_l, x_u, th.tensor([[0, 1, 0]]))
        self.assertFalse(all_bad)                     # action 1 is not the greedy action
        # brute-force sanity inside the box
        pts = th.tensor(np.random.uniform([-0.1, -0.1], [0.1, 0.1], size=(500, 2)), dtype=th.float32)
        self.assertTrue(bool((seq(pts).argmax(1) == 0).all()))


class DiscreteSafeInitTests(unittest.TestCase):
    def test_dqn_pretrain_certifies_and_predicts_safe(self) -> None:
        mask = _only_action_safe(1)
        model = ProvablySafeDQN(
            "MlpPolicy", gym.make("FrozenLake-v1"), device="cpu", shield=mask, seed=0,
            shield_seed=0, learning_starts=20, buffer_size=300, batch_size=8,
            policy_kwargs={"net_arch": [16]},
        )
        self.addCleanup(model.get_env().close)
        report = model.pretrain_on_shield(lr=5e-2, bc_max_epochs=300, refine_max_epochs=300,
                                          require_certified=True)
        self.assertEqual(report.sampled_greedy_safe_rate, 1.0)
        self.assertTrue(report.all_certified)
        # Unshielded greedy predictions are all the safe action.
        actions = [int(np.asarray(model.predict(np.array([s]), deterministic=True)[0]).item())
                   for s in range(16)]
        self.assertTrue(all(a == 1 for a in actions))

    def test_ppo_pretrain_certifies(self) -> None:
        mask = _only_action_safe(2)
        model = ProvablySafePPO(
            "MlpPolicy", gym.make("FrozenLake-v1"), device="cpu", shield=mask, seed=0,
            shield_seed=0, n_steps=32, batch_size=16, n_epochs=1, policy_kwargs={"net_arch": [16]},
        )
        self.addCleanup(model.get_env().close)
        report = model.pretrain_on_shield(lr=5e-2, bc_max_epochs=300, refine_max_epochs=300,
                                          require_certified=True)
        self.assertEqual(report.sampled_greedy_safe_rate, 1.0)
        self.assertTrue(report.all_certified)


class ContinuousSafeInitTests(unittest.TestCase):
    def _make_dqn(self, shield) -> ProvablySafeDQN:
        model = ProvablySafeDQN(
            "MlpPolicy", gym.make("MountainCar-v0"), device="cpu", shield=shield, seed=0,
            shield_seed=0, learning_starts=20, buffer_size=300, batch_size=8,
            policy_kwargs={"net_arch": [16]},
        )
        self.addCleanup(model.get_env().close)
        return model

    def test_dqn_box_shield_certifies(self) -> None:
        model = self._make_dqn(_mc_box_shield())
        report = model.pretrain_on_shield(n_samples=1024, lr=5e-2, bc_max_epochs=300,
                                          refine_max_epochs=500, require_certified=True)
        self.assertTrue(report.all_certified)
        self.assertEqual(report.certified_fraction, 1.0)
        self.assertGreaterEqual(report.sampled_greedy_safe_rate, 0.99)

    def test_ppo_box_shield_certifies(self) -> None:
        model = ProvablySafePPO(
            "MlpPolicy", gym.make("MountainCar-v0"), device="cpu", shield=_mc_box_shield(),
            seed=0, shield_seed=0, n_steps=64, policy_kwargs={"net_arch": [16]},
        )
        self.addCleanup(model.get_env().close)
        report = model.pretrain_on_shield(n_samples=1024, lr=5e-2, bc_max_epochs=300,
                                          refine_max_epochs=500, require_certified=True)
        self.assertTrue(report.all_certified)

    def test_require_certified_raises_when_uncertified(self) -> None:
        model = self._make_dqn(_mc_box_shield())
        # No training budget -> the fresh network is not certified -> raise.
        with self.assertRaises(RuntimeError):
            model.pretrain_on_shield(bc_max_epochs=0, refine_max_epochs=0, require_certified=True)

    def test_predicate_shield_skips_certification(self) -> None:
        shield = RegionShield(regions=[(lambda o: o[0] < 0.0, [2])], n_actions=3, seed=0)
        model = self._make_dqn(shield)
        with self.assertWarns(UserWarning):
            report = model.pretrain_on_shield(n_samples=512, lr=5e-2, bc_max_epochs=300)
        self.assertIsNone(report.certified_fraction)
        self.assertGreaterEqual(report.sampled_greedy_safe_rate, 0.99)


if __name__ == "__main__":
    unittest.main()
