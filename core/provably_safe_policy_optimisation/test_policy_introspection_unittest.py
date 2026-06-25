"""Tests for SB3 policy introspection helpers."""

from __future__ import annotations

import unittest

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO

from provably_safe_policy_optimisation.policy_introspection import (
    extract_feature_actor_parameters_and_network,
    resolve_feature_actor_names_for_policy,
    resolve_policy,
)


def _make_ppo() -> PPO:
    return PPO(
        "MlpPolicy",
        gym.make("CartPole-v1"),
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": [16]},
    )


class PolicyIntrospectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _make_ppo()
        self.addCleanup(self.model.get_env().close)

    def test_resolve_policy_accepts_model_and_policy(self) -> None:
        self.assertIs(resolve_policy(self.model), self.model.policy)
        self.assertIs(resolve_policy(self.model.policy), self.model.policy)

    def test_resolve_policy_rejects_non_policy(self) -> None:
        with self.assertRaises(TypeError):
            resolve_policy(object())

    def test_feature_actor_names_are_actor_only_and_sorted(self) -> None:
        name_to_param = dict(self.model.policy.named_parameters())
        names = resolve_feature_actor_names_for_policy(self.model.policy, name_to_param)
        self.assertTrue(names)
        self.assertEqual(names, sorted(names))
        self.assertTrue(set(names).issubset(name_to_param))
        # Actor branch must not include the value network.
        self.assertFalse(any("value_net" in n for n in names))
        self.assertTrue(any("action_net" in n for n in names))

    def test_extract_network_runs_and_matches_param_set(self) -> None:
        params, network = extract_feature_actor_parameters_and_network(self.model)
        self.assertIsInstance(network, th.nn.Sequential)
        # Returned parameter names match the resolved feature-actor names.
        name_to_param = dict(self.model.policy.named_parameters())
        expected = set(resolve_feature_actor_names_for_policy(self.model.policy, name_to_param))
        self.assertEqual(set(params), expected)
        # The extracted actor network produces one logit per discrete action.
        obs = th.as_tensor(
            self.model.observation_space.sample(), dtype=th.float32
        ).unsqueeze(0)
        with th.no_grad():
            out = network(obs)
        self.assertEqual(out.shape[-1], int(self.model.action_space.n))


if __name__ == "__main__":
    unittest.main()
