"""Integration tests for continuous-state shielding (RegionShield) on MountainCar-v0."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from provably_safe_policy_optimisation import ProvablySafeDQN, ProvablySafePPO, RegionShield


def _mountaincar_shield() -> RegionShield:
    # When position < 0.3, only "push right" (action 2) is safe; elsewhere all safe.
    return RegionShield(regions=[(lambda o: o[0] < 0.3, [2])], n_actions=3, seed=0)


class _RecordObsActions(gym.Wrapper):
    """Records (observation_before, action_executed) for every env step."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.records: list[tuple[np.ndarray, int]] = []
        self._last: np.ndarray | None = None

    def reset(self, **kwargs):  # type: ignore[no-untyped-def]
        obs, info = self.env.reset(**kwargs)
        self._last = np.asarray(obs, dtype=np.float64)
        return obs, info

    def step(self, action):  # type: ignore[no-untyped-def]
        self.records.append((self._last, int(action)))
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last = np.asarray(obs, dtype=np.float64)
        return obs, reward, terminated, truncated, info


def _all_actions_safe(shield: RegionShield, records) -> bool:
    for obs, action in records:
        state = int(shield.obs_to_state(obs.reshape(1, -1))[0])
        if shield.mask[state, action] != 1:
            return False
    return True


class ContinuousShieldDQNTests(unittest.TestCase):
    def test_executed_actions_safe(self) -> None:
        shield = _mountaincar_shield()
        recorder = _RecordObsActions(gym.make("MountainCar-v0"))
        model = ProvablySafeDQN(
            "MlpPolicy", recorder, shield=shield, seed=0, shield_seed=0, device="cpu", verbose=0,
            learning_starts=20, buffer_size=500, batch_size=8, train_freq=1,
            target_update_interval=10, policy_kwargs={"net_arch": [16]},
        )
        self.addCleanup(model.get_env().close)
        model.learn(total_timesteps=300)
        self.assertTrue(recorder.records)
        self.assertTrue(_all_actions_safe(shield, recorder.records))
        self.assertGreater(model.shield_diagnostics()["overridden"], 0)


class ContinuousShieldPPOTests(unittest.TestCase):
    def test_executed_actions_safe(self) -> None:
        shield = _mountaincar_shield()
        recorder = _RecordObsActions(gym.make("MountainCar-v0"))
        model = ProvablySafePPO(
            "MlpPolicy", recorder, shield=shield, seed=0, shield_seed=0, device="cpu", verbose=0,
            n_steps=64, batch_size=32, n_epochs=1, policy_kwargs={"net_arch": [16]},
        )
        self.addCleanup(model.get_env().close)
        model.learn(total_timesteps=128)
        self.assertTrue(recorder.records)
        self.assertTrue(_all_actions_safe(shield, recorder.records))
        self.assertGreater(model.shield_diagnostics()["overridden"], 0)

    def test_predict_not_shielded(self) -> None:
        model = ProvablySafePPO(
            "MlpPolicy", gym.make("MountainCar-v0"), shield=_mountaincar_shield(),
            seed=0, shield_seed=0, device="cpu", verbose=0,
            n_steps=64, batch_size=32, n_epochs=1, policy_kwargs={"net_arch": [16]},
        )
        self.addCleanup(model.get_env().close)
        before = model.shield_diagnostics()["checked"]
        for _ in range(10):
            model.predict(np.array([-0.5, 0.0]), deterministic=False)
        self.assertEqual(model.shield_diagnostics()["checked"], before)


if __name__ == "__main__":
    unittest.main()
