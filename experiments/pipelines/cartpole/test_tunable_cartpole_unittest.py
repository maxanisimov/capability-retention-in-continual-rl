"""Unit tests for TunableCartPoleEnv."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import gymnasium as gym

    from experiments.pipelines.cartpole.core.env import (
        TUNABLE_CARTPOLE_V1_ID,
        TunableCartPoleEnv,
        make_cartpole_env,
    )
except Exception as exc:  # pragma: no cover - dependency guard
    gym = None  # type: ignore[assignment]
    TUNABLE_CARTPOLE_V1_ID = "TunableCartPole-v1"
    TunableCartPoleEnv = None  # type: ignore[assignment]
    make_cartpole_env = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_ENV_AVAILABLE = bool(gym is not None and TunableCartPoleEnv is not None)


@unittest.skipUnless(
    _ENV_AVAILABLE,
    f"Gymnasium/TunableCartPole dependencies unavailable: {_IMPORT_ERROR!r}",
)
class TunableCartPoleTests(unittest.TestCase):
    def test_gym_make_accepts_tunable_physics(self):
        env = gym.make(
            TUNABLE_CARTPOLE_V1_ID,
            gravity=12.5,
            masscart=1.2,
            masspole=0.2,
            length=0.7,
            force_mag=8.0,
        )
        try:
            base = env.unwrapped
            self.assertIsInstance(base, TunableCartPoleEnv)
            self.assertEqual(base.gravity, 12.5)
            self.assertEqual(base.masscart, 1.2)
            self.assertEqual(base.masspole, 0.2)
            self.assertEqual(base.length, 0.7)
            self.assertEqual(base.force_mag, 8.0)
            self.assertAlmostEqual(base.total_mass, 1.4)
            self.assertAlmostEqual(base.polemass_length, 0.14)
        finally:
            env.close()

    def test_observation_space_bounds_update(self):
        env = gym.make(
            TUNABLE_CARTPOLE_V1_ID,
            x_threshold=3.0,
            theta_threshold_radians=0.4,
        )
        try:
            np.testing.assert_allclose(
                env.unwrapped.observation_space.high[[0, 2]],
                np.asarray([6.0, 0.8], dtype=np.float32),
            )
        finally:
            env.close()

    def test_reset_bounds_and_options_override(self):
        env = gym.make(TUNABLE_CARTPOLE_V1_ID, reset_low=-0.02, reset_high=0.03)
        try:
            for seed in range(5):
                obs, _ = env.reset(seed=seed)
                self.assertTrue(np.all(obs >= -0.02))
                self.assertTrue(np.all(obs <= 0.03))

            obs, _ = env.reset(seed=123, options={"low": 0.01, "high": 0.02})
            self.assertTrue(np.all(obs >= 0.01))
            self.assertTrue(np.all(obs <= 0.02))
        finally:
            env.close()

    def test_invalid_settings_raise(self):
        invalid_kwargs = [
            {"gravity": float("nan")},
            {"masscart": 0.0},
            {"masspole": -0.1},
            {"length": 0.0},
            {"force_mag": 0.0},
            {"tau": -0.01},
            {"theta_threshold_radians": 0.0},
            {"x_threshold": -2.4},
            {"kinematics_integrator": "bad"},
            {"reset_low": 0.1, "reset_high": -0.1},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    env = gym.make(TUNABLE_CARTPOLE_V1_ID, **kwargs)
                    env.close()

    def test_factory_appends_task_id(self):
        env = make_cartpole_env(task_id=math.pi, append_task_id=True)
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (5,))
            self.assertAlmostEqual(float(obs[-1]), math.pi, places=6)
            self.assertEqual(env.observation_space.shape, (5,))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
