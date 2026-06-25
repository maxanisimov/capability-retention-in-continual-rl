"""Integration tests for ProvablySafeDQN and ProvablySafePPO on FrozenLake-v1."""

from __future__ import annotations

import unittest
import warnings

import gymnasium as gym
import numpy as np
import torch as th

from provably_safe_policy_optimisation import ProvablySafeDQN, ProvablySafePPO


def _only_action_safe(action: int, n_states: int = 16, n_actions: int = 4) -> np.ndarray:
    """A shield where only ``action`` is safe in every state."""
    mask = np.zeros((n_states, n_actions), dtype=int)
    mask[:, action] = 1
    return mask


class _RecordActions(gym.Wrapper):
    """Records (state_before, action_executed) for every env step."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.records: list[tuple[int, int]] = []
        self._last: int = 0

    def reset(self, **kwargs):  # type: ignore[no-untyped-def]
        obs, info = self.env.reset(**kwargs)
        self._last = int(obs)
        return obs, info

    def step(self, action):  # type: ignore[no-untyped-def]
        self.records.append((self._last, int(action)))
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last = int(obs)
        return obs, reward, terminated, truncated, info


class ProvablySafeDQNTests(unittest.TestCase):
    def _make(self, mask, recorder=None, **extra) -> ProvablySafeDQN:
        env = recorder or gym.make("FrozenLake-v1")
        model = ProvablySafeDQN(
            "MlpPolicy", env, shield=mask, seed=0, shield_seed=0, device="cpu", verbose=0,
            learning_starts=20, buffer_size=500, batch_size=8, train_freq=1,
            target_update_interval=10, policy_kwargs={"net_arch": [16]}, **extra,
        )
        self.addCleanup(model.get_env().close)
        return model

    def test_action_space_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make(_only_action_safe(0, n_actions=3))  # 3 != 4

    def test_sample_action_is_always_safe(self) -> None:
        mask = _only_action_safe(1)
        model = self._make(mask)
        model._last_obs = np.array([5])
        for _ in range(30):
            action, buffer = model._sample_action(learning_starts=0, n_envs=1)
            self.assertEqual(int(action[0]), 1)
            self.assertTrue(np.array_equal(action, buffer))

    def test_every_executed_action_is_safe_during_training(self) -> None:
        mask = _only_action_safe(1)
        recorder = _RecordActions(gym.make("FrozenLake-v1"))
        model = self._make(mask, recorder=recorder)
        model.learn(total_timesteps=200)
        self.assertTrue(recorder.records)
        self.assertTrue(all(mask[s, a] == 1 for s, a in recorder.records))
        self.assertGreater(model.shield_diagnostics()["overridden"], 0)

    def test_predict_is_not_shielded(self) -> None:
        model = self._make(_only_action_safe(1))
        before = model.shield_diagnostics()["checked"]
        for _ in range(10):
            model.predict(np.array([3]), deterministic=True)
        self.assertEqual(model.shield_diagnostics()["checked"], before)

    def test_shielding_and_projection_compose(self) -> None:
        mask = _only_action_safe(1)
        probe = self._make(mask)
        params = list(probe.policy.q_net.parameters())
        lower = [p.detach() - 0.02 for p in params]
        upper = [p.detach() + 0.02 for p in params]
        model = self._make(mask, learning_rate=1e-2, param_bounds_l=lower, param_bounds_u=upper)
        model.learn(total_timesteps=200)
        self.assertTrue(model.is_within_bounds(atol=1e-6))
        self.assertGreater(model.shield_diagnostics()["overridden"], 0)


class ProvablySafePPOTests(unittest.TestCase):
    def _make(self, mask, recorder=None, **extra) -> ProvablySafePPO:
        env = recorder or gym.make("FrozenLake-v1")
        model = ProvablySafePPO(
            "MlpPolicy", env, shield=mask, seed=0, shield_seed=0, device="cpu", verbose=0,
            n_steps=32, batch_size=16, n_epochs=1, policy_kwargs={"net_arch": [16]}, **extra,
        )
        self.addCleanup(model.get_env().close)
        return model

    def test_wrapped_forward_returns_safe_consistent_actions(self) -> None:
        mask = _only_action_safe(1)
        model = self._make(mask)
        obs_tensor, _ = model.policy.obs_to_tensor(np.array([0, 5, 10]))
        with th.no_grad():
            actions, _values, log_prob = model.policy(obs_tensor)
            self.assertTrue(th.all(actions == 1))                      # all shielded to safe action
            expected = model.policy.get_distribution(obs_tensor).log_prob(actions)
        self.assertTrue(th.allclose(log_prob, expected))               # consistent log-prob

    def test_every_executed_action_is_safe_during_training(self) -> None:
        mask = _only_action_safe(1)
        recorder = _RecordActions(gym.make("FrozenLake-v1"))
        model = self._make(mask, recorder=recorder)
        model.learn(total_timesteps=64)
        self.assertTrue(recorder.records)
        self.assertTrue(all(mask[s, a] == 1 for s, a in recorder.records))
        self.assertGreater(model.shield_diagnostics()["overridden"], 0)

    def test_predict_is_not_shielded(self) -> None:
        model = self._make(_only_action_safe(1))
        before = model.shield_diagnostics()["checked"]
        for _ in range(10):
            model.predict(np.array([3]), deterministic=False)
        self.assertEqual(model.shield_diagnostics()["checked"], before)


if __name__ == "__main__":
    unittest.main()
