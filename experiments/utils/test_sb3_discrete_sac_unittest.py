"""Unit tests for local SB3-compatible Discrete SAC."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from experiments.utils.sb3_discrete_sac import DiscreteSAC


class DiscreteSACTests(unittest.TestCase):
    def _make_model(self) -> DiscreteSAC:
        env = gym.make("CartPole-v1")
        model = DiscreteSAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=2_000,
            learning_starts=16,
            batch_size=32,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            tau=0.01,
            seed=7,
            device="cpu",
            policy_kwargs={"net_arch": [64, 64]},
        )
        self.addCleanup(model.get_env().close)
        return model

    def test_predict_returns_valid_discrete_action(self) -> None:
        model = self._make_model()
        env = model.get_env()
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=False)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (1,))
        self.assertGreaterEqual(int(action[0]), 0)
        self.assertLess(int(action[0]), int(model.action_space.n))

    def test_short_training_runs_and_updates(self) -> None:
        model = self._make_model()
        model.learn(total_timesteps=128, progress_bar=False)
        self.assertGreaterEqual(model.num_timesteps, 128)
        self.assertGreater(model._n_updates, 0)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
