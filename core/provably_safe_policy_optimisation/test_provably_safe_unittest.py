"""Integration tests for ProvablySafeDQN and ProvablySafePPO on FrozenLake-v1."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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

    def _force_policy_to_propose(self, model: ProvablySafePPO, action: int) -> None:
        def forward(obs, deterministic: bool = False):  # type: ignore[no-untyped-def]
            del deterministic
            actions = th.full((obs.shape[0],), action, dtype=th.long, device=obs.device)
            values = model.policy.predict_values(obs)
            log_prob = model.policy.get_distribution(obs).log_prob(actions)
            return actions, values, log_prob

        model.policy.forward = forward  # type: ignore[method-assign]

    def test_policy_forward_is_not_shielded(self) -> None:
        mask = _only_action_safe(1)
        model = self._make(mask)
        self._force_policy_to_propose(model, action=0)
        obs_tensor, _ = model.policy.obs_to_tensor(np.array([0, 5, 10]))
        with th.no_grad():
            actions, _values, log_prob = model.policy(obs_tensor)
            self.assertTrue(th.all(actions == 0))
            expected = model.policy.get_distribution(obs_tensor).log_prob(actions)
        self.assertTrue(th.allclose(log_prob, expected))

    def test_default_stores_proposed_actions_but_executes_safe_actions(self) -> None:
        mask = _only_action_safe(1)
        recorder = _RecordActions(gym.make("FrozenLake-v1"))
        model = self._make(mask, recorder=recorder)
        self._force_policy_to_propose(model, action=0)
        model.learn(total_timesteps=32)
        self.assertTrue(recorder.records)
        self.assertTrue(all(mask[s, a] == 1 for s, a in recorder.records))
        self.assertTrue(np.all(model.rollout_buffer.actions == 0))

    def test_executed_storage_mode_stores_overridden_actions(self) -> None:
        mask = _only_action_safe(1)
        recorder = _RecordActions(gym.make("FrozenLake-v1"))
        model = self._make(mask, recorder=recorder, shield_action_storage="executed")
        self._force_policy_to_propose(model, action=0)
        model.learn(total_timesteps=32)
        self.assertTrue(recorder.records)
        self.assertTrue(all(mask[s, a] == 1 for s, a in recorder.records))
        self.assertTrue(np.all(model.rollout_buffer.actions == 1))

    def test_unsafe_action_callback_counts_raw_policy_proposals(self) -> None:
        mask = _only_action_safe(1)
        model = self._make(mask)
        self._force_policy_to_propose(model, action=0)
        events = []
        model.set_exploration_unsafe_action_callback(lambda **row: events.append(row))

        model.learn(total_timesteps=32)

        self.assertTrue(events)
        self.assertEqual(events[-1]["timestep"], 32)
        self.assertTrue(all(event["checked_this_step"] == 1 for event in events))
        self.assertTrue(all(event["unsafe_this_step"] == 1 for event in events))
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(Path(tmpdir) / "model.zip")

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
