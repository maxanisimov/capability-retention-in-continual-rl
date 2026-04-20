"""Unit tests for local SB3 PPO parameter clamping."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

from experiments.utils.sb3_clamped_ppo import (
    ClampedPPO,
    extract_feature_actor_parameters_and_network,
)


class TinyTrainableExtractor(BaseFeaturesExtractor):
    """Minimal trainable feature extractor for selector tests."""

    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 8):
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.linear = nn.Linear(in_dim, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return th.tanh(self.linear(observations))


class TinyNetWrappedExtractor(BaseFeaturesExtractor):
    """Feature extractor that wraps its stack under ``self.net``."""

    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 8):
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


def _is_all_zero(tensor: th.Tensor) -> bool:
    return bool(th.allclose(tensor.detach(), th.zeros_like(tensor), atol=1e-8, rtol=0.0))


class ClampedPPOTests(unittest.TestCase):
    def _feature_actor_names(self, model: ClampedPPO) -> list[str]:
        params = dict(model.policy.named_parameters())
        return sorted(
            name
            for name in params
            if name.startswith(("features_extractor.", "mlp_extractor.policy_net.", "action_net."))
        )

    def _make_model(
        self,
        *,
        env_id: str,
        rules: list[dict[str, object]],
        share_features_extractor: bool = True,
    ) -> ClampedPPO:
        env = gym.make(env_id)
        model = ClampedPPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            learning_rate=3e-4,
            seed=7,
            device="cpu",
            policy_kwargs={
                "features_extractor_class": TinyTrainableExtractor,
                "features_extractor_kwargs": {"features_dim": 8},
                "share_features_extractor": share_features_extractor,
                "net_arch": {"pi": [8], "vf": [8]},
            },
            param_clamp_rules=rules,
        )
        self.addCleanup(model.get_env().close)
        return model

    def _assert_init_raises(
        self,
        *,
        env_id: str,
        rules: list[dict[str, object]],
        message_regex: str,
    ) -> None:
        env = gym.make(env_id)
        model: ClampedPPO | None = None
        try:
            with self.assertRaisesRegex(ValueError, message_regex):
                model = ClampedPPO(
                    "MlpPolicy",
                    env,
                    n_steps=8,
                    batch_size=8,
                    n_epochs=1,
                    learning_rate=3e-4,
                    seed=7,
                    device="cpu",
                    policy_kwargs={
                        "features_extractor_class": TinyTrainableExtractor,
                        "features_extractor_kwargs": {"features_dim": 8},
                        "share_features_extractor": True,
                        "net_arch": {"pi": [8], "vf": [8]},
                    },
                    param_clamp_rules=rules,
                )
        finally:
            if model is not None:
                model.get_env().close()
            env.close()

    def test_feature_actor_clamps_actor_and_features_only(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[{"selector": "feature_actor", "min_value": 0.0, "max_value": 0.0}],
            share_features_extractor=True,
        )
        model.learn(total_timesteps=32, progress_bar=False)
        params = dict(model.policy.named_parameters())

        for name, param in params.items():
            if name.startswith(("features_extractor.", "mlp_extractor.policy_net.", "action_net.")):
                self.assertTrue(_is_all_zero(param), msg=f"{name} expected clamped to zero")
            if name.startswith(("mlp_extractor.value_net.", "value_net.")):
                self.assertFalse(_is_all_zero(param), msg=f"{name} should not be clamped by feature_actor")

        self.assertGreater(model.param_clamp_step_calls, 0)
        self.assertGreater(model.param_clamp_projected_elements, 0)

    def test_feature_critic_clamps_critic_and_features_only(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[{"selector": "feature_critic", "min_value": 0.0, "max_value": 0.0}],
            share_features_extractor=True,
        )
        model.learn(total_timesteps=32, progress_bar=False)
        params = dict(model.policy.named_parameters())

        for name, param in params.items():
            if name.startswith(("features_extractor.", "mlp_extractor.value_net.", "value_net.")):
                self.assertTrue(_is_all_zero(param), msg=f"{name} expected clamped to zero")
            if name.startswith(("mlp_extractor.policy_net.", "action_net.")):
                self.assertFalse(_is_all_zero(param), msg=f"{name} should not be clamped by feature_critic")

    def test_shared_features_overlap_allowed_if_bounds_identical(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[
                {"selector": "feature_actor", "min_value": 0.0, "max_value": 0.0},
                {"selector": "feature_critic", "min_value": 0.0, "max_value": 0.0},
            ],
            share_features_extractor=True,
        )
        model.learn(total_timesteps=32, progress_bar=False)
        params = dict(model.policy.named_parameters())

        for name, param in params.items():
            if name.startswith(
                (
                    "features_extractor.",
                    "mlp_extractor.policy_net.",
                    "mlp_extractor.value_net.",
                    "action_net.",
                    "value_net.",
                ),
            ):
                self.assertTrue(_is_all_zero(param), msg=f"{name} expected clamped to zero")

    def test_shared_features_overlap_rejected_if_bounds_differ(self) -> None:
        self._assert_init_raises(
            env_id="CartPole-v1",
            rules=[
                {"selector": "feature_actor", "min_value": 0.0, "max_value": 0.0},
                {"selector": "feature_critic", "min_value": -1.0, "max_value": 1.0},
            ],
            message_regex="Overlapping clamp rules",
        )

    def test_non_shared_feature_extractors_are_clamped_independently(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[{"selector": "feature_actor", "min_value": 0.0, "max_value": 0.0}],
            share_features_extractor=False,
        )
        model.learn(total_timesteps=32, progress_bar=False)
        params = dict(model.policy.named_parameters())

        for name, param in params.items():
            if name.startswith(("features_extractor.", "mlp_extractor.policy_net.", "action_net.")):
                self.assertTrue(_is_all_zero(param), msg=f"{name} expected clamped to zero")
            if name.startswith(("vf_features_extractor.", "mlp_extractor.value_net.", "value_net.")):
                self.assertFalse(_is_all_zero(param), msg=f"{name} should not be clamped by feature_actor")

    def test_strict_validation_checks(self) -> None:
        self._assert_init_raises(
            env_id="CartPole-v1",
            rules=[{"selector": "actor_head", "min_value": 1.0, "max_value": -1.0}],
            message_regex="invalid bounds",
        )
        self._assert_init_raises(
            env_id="CartPole-v1",
            rules=[{"selector": "unsupported_selector", "min_value": 0.0, "max_value": 0.0}],
            message_regex="unsupported selector",
        )
        self._assert_init_raises(
            env_id="CartPole-v1",
            rules=[{"name_prefix": "no_such_parameter_prefix.", "min_value": 0.0, "max_value": 0.0}],
            message_regex="matched no parameters",
        )
        self._assert_init_raises(
            env_id="CartPole-v1",
            rules=[
                {"selector": "actor_head", "min_value": 0.0, "max_value": 0.0},
                {"name_prefix": "action_net.", "min_value": 0.0, "max_value": 0.0},
            ],
            message_regex="Overlapping clamp rules",
        )

    def test_log_std_selector_on_continuous_policy(self) -> None:
        model = self._make_model(
            env_id="Pendulum-v1",
            rules=[{"selector": "log_std", "min_value": 0.0, "max_value": 0.0}],
            share_features_extractor=True,
        )
        model.learn(total_timesteps=32, progress_bar=False)
        params = dict(model.policy.named_parameters())
        self.assertIn("log_std", params)
        self.assertTrue(_is_all_zero(params["log_std"]))

    def test_set_feature_actor_parameter_bounds_with_sequence(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[],
            share_features_extractor=True,
        )
        params = dict(model.policy.named_parameters())
        names = self._feature_actor_names(model)
        lower = [th.full_like(params[name], 0.0) for name in names]
        upper = [th.full_like(params[name], 0.0) for name in names]
        resolved = model.set_feature_actor_parameter_bounds(lower, upper)
        self.assertEqual(resolved, tuple(names))

        model.learn(total_timesteps=32, progress_bar=False)
        for name in names:
            self.assertTrue(_is_all_zero(params[name]), msg=f"{name} expected clamped to zero")

    def test_set_feature_actor_parameter_bounds_with_mapping(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[],
            share_features_extractor=True,
        )
        params = dict(model.policy.named_parameters())
        names = self._feature_actor_names(model)
        lower = {name: th.full_like(params[name], -0.01) for name in names}
        upper = {name: th.full_like(params[name], 0.01) for name in names}
        resolved = model.set_feature_actor_parameter_bounds(lower, upper)
        self.assertEqual(resolved, tuple(names))

        model.learn(total_timesteps=32, progress_bar=False)
        for name in names:
            tensor = params[name].detach()
            self.assertTrue(bool(th.all(tensor >= -0.01)))
            self.assertTrue(bool(th.all(tensor <= 0.01)))

    def test_set_feature_actor_parameter_bounds_validation(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[],
            share_features_extractor=True,
        )
        params = dict(model.policy.named_parameters())
        names = self._feature_actor_names(model)

        with self.assertRaisesRegex(ValueError, "Sequence bounds must have length"):
            model.set_feature_actor_parameter_bounds(
                [th.full_like(params[names[0]], 0.0)],
                [th.full_like(params[names[0]], 0.0)],
            )

        lower_map = {name: th.full_like(params[name], 0.0) for name in names}
        upper_map = {name: th.full_like(params[name], 0.0) for name in names}
        bad_upper_map = dict(upper_map)
        bad_upper_map.pop(names[0])
        with self.assertRaisesRegex(ValueError, "must exactly match"):
            model.set_feature_actor_parameter_bounds(lower_map, bad_upper_map)

        bad_lower = [th.full_like(params[name], 1.0) for name in names]
        bad_upper = [th.full_like(params[name], 0.0) for name in names]
        with self.assertRaisesRegex(ValueError, "lower has values greater than upper"):
            model.set_feature_actor_parameter_bounds(bad_lower, bad_upper)

    def test_extract_feature_actor_parameters_and_network_shared(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[],
            share_features_extractor=True,
        )
        params, network = extract_feature_actor_parameters_and_network(model)
        names = self._feature_actor_names(model)
        self.assertEqual(sorted(params.keys()), names)

        obs = th.randn(6, 4, dtype=th.float32)
        with th.no_grad():
            policy = model.policy
            feats = policy.features_extractor(obs)
            expected = policy.action_net(policy.mlp_extractor.policy_net(feats))
            got = network(obs)
        self.assertTrue(bool(th.allclose(got, expected, atol=1e-6, rtol=1e-5)))

    def test_extract_feature_actor_parameters_and_network_non_shared(self) -> None:
        model = self._make_model(
            env_id="CartPole-v1",
            rules=[],
            share_features_extractor=False,
        )
        params, network = extract_feature_actor_parameters_and_network(model)
        names = self._feature_actor_names(model)
        self.assertEqual(sorted(params.keys()), names)

        obs = th.randn(6, 4, dtype=th.float32)
        with th.no_grad():
            policy = model.policy
            feats = policy.pi_features_extractor(obs)
            expected = policy.action_net(policy.mlp_extractor.policy_net(feats))
            got = network(obs)
        self.assertTrue(bool(th.allclose(got, expected, atol=1e-6, rtol=1e-5)))

    def test_extract_feature_actor_parameters_and_network_unwraps_layers(self) -> None:
        env = gym.make("CartPole-v1")
        model = ClampedPPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            learning_rate=3e-4,
            seed=7,
            device="cpu",
            policy_kwargs={
                "features_extractor_class": TinyNetWrappedExtractor,
                "features_extractor_kwargs": {"features_dim": 8},
                "share_features_extractor": True,
                "net_arch": {"pi": [8, 8], "vf": [8]},
            },
            param_clamp_rules=[],
        )
        self.addCleanup(model.get_env().close)

        _, network = extract_feature_actor_parameters_and_network(model)
        modules = list(network.children())

        self.assertFalse(any(isinstance(module, TinyNetWrappedExtractor) for module in modules))
        self.assertFalse(any(isinstance(module, nn.Sequential) for module in modules))
        self.assertGreaterEqual(len(modules), 6)
        self.assertIsInstance(modules[0], nn.Flatten)
        self.assertIsInstance(modules[-1], nn.Linear)

        obs = th.randn(6, 4, dtype=th.float32)
        with th.no_grad():
            policy = model.policy
            expected = policy.action_net(policy.mlp_extractor.policy_net(policy.features_extractor(obs)))
            got = network(obs)
        self.assertTrue(bool(th.allclose(got, expected, atol=1e-6, rtol=1e-5)))

    def test_extract_feature_actor_parameters_and_network_unwraps_flatten_extractor(self) -> None:
        env = gym.make("CartPole-v1")
        model = ClampedPPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            learning_rate=3e-4,
            seed=7,
            device="cpu",
            policy_kwargs={
                "features_extractor_class": FlattenExtractor,
                "share_features_extractor": True,
                "net_arch": {"pi": [8], "vf": [8]},
            },
            param_clamp_rules=[],
        )
        self.addCleanup(model.get_env().close)

        _, network = extract_feature_actor_parameters_and_network(model)
        modules = list(network.children())
        self.assertFalse(any(isinstance(module, FlattenExtractor) for module in modules))
        self.assertIsInstance(modules[0], nn.Flatten)

        obs = th.randn(6, 4, dtype=th.float32)
        with th.no_grad():
            policy = model.policy
            expected = policy.action_net(policy.mlp_extractor.policy_net(policy.features_extractor(obs)))
            got = network(obs)
        self.assertTrue(bool(th.allclose(got, expected, atol=1e-6, rtol=1e-5)))


if __name__ == "__main__":
    unittest.main()
