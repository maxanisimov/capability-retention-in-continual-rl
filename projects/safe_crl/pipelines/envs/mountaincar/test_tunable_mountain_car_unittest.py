"""Unit tests for TunableMountainCarEnv."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - dependency guard
    gym = None  # type: ignore[assignment]

try:
    from projects.safe_crl.pipelines.envs.mountaincar.core.env.env_factory import make_mountaincar_env
    from projects.safe_crl.pipelines.envs.mountaincar.core.env.tunable_mountain_car import (
        TUNABLE_MOUNTAIN_CAR_V0_ID,
        TunableMountainCarEnv,
    )

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency guard
    make_mountaincar_env = None  # type: ignore[assignment]
    TUNABLE_MOUNTAIN_CAR_V0_ID = "TunableMountainCar-v0"
    TunableMountainCarEnv = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


_ENV_AVAILABLE = bool(gym is not None and TunableMountainCarEnv is not None)


@unittest.skipUnless(
    _ENV_AVAILABLE,
    f"Gymnasium/TunableMountainCar dependencies unavailable: {_IMPORT_ERROR!r}",
)
class TunableMountainCarTests(unittest.TestCase):
    def test_gym_make_stores_tunable_attributes(self) -> None:
        env = gym.make(
            TUNABLE_MOUNTAIN_CAR_V0_ID,
            force=0.002,
            gravity=0.003,
            goal_position=0.45,
        )
        try:
            base = env.unwrapped
            self.assertIsInstance(base, TunableMountainCarEnv)
            self.assertAlmostEqual(base.force, 0.002)
            self.assertAlmostEqual(base.gravity, 0.003)
            self.assertAlmostEqual(base.goal_position, 0.45)
        finally:
            env.close()

    def test_observation_space_bounds_follow_position_and_speed_settings(self) -> None:
        env = gym.make(
            TUNABLE_MOUNTAIN_CAR_V0_ID,
            min_position=-1.5,
            max_position=0.8,
            max_speed=0.09,
            goal_position=0.7,
        )
        try:
            np.testing.assert_allclose(
                env.observation_space.low,
                np.array([-1.5, -0.09], dtype=np.float32),
            )
            np.testing.assert_allclose(
                env.observation_space.high,
                np.array([0.8, 0.09], dtype=np.float32),
            )
        finally:
            env.close()

    def test_reset_uses_configured_bounds_and_options_override_them(self) -> None:
        env = gym.make(
            TUNABLE_MOUNTAIN_CAR_V0_ID,
            reset_low=-0.3,
            reset_high=-0.2,
        )
        try:
            obs, _ = env.reset(seed=123)
            self.assertGreaterEqual(float(obs[0]), -0.3)
            self.assertLessEqual(float(obs[0]), -0.2)
            self.assertEqual(float(obs[1]), 0.0)

            obs, _ = env.reset(seed=123, options={"low": -0.1, "high": 0.1})
            self.assertGreaterEqual(float(obs[0]), -0.1)
            self.assertLessEqual(float(obs[0]), 0.1)
            self.assertEqual(float(obs[1]), 0.0)
        finally:
            env.close()

    def test_goal_reaching_step_reports_success_and_safe(self) -> None:
        env = gym.make(TUNABLE_MOUNTAIN_CAR_V0_ID)
        try:
            env.reset(seed=0)
            base = env.unwrapped
            base.state = np.array(
                [base.goal_position, base.goal_velocity],
                dtype=np.float64,
            )

            _, _, terminated, _, info = env.step(2)

            self.assertTrue(terminated)
            self.assertIn("is_success", info)
            self.assertIs(info["is_success"], True)
            self.assertIn("safe", info)
            self.assertIs(info["safe"], True)
        finally:
            env.close()

    def test_invalid_settings_raise_value_error(self) -> None:
        invalid_kwargs = [
            {"force": -0.001},
            {"gravity": -0.0025},
            {"force": np.nan},
            {"gravity": np.inf},
            {"min_position": 0.0, "max_position": 0.0},
            {"max_speed": 0.0},
            {"goal_position": 2.0},
            {"reset_low": 0.2, "reset_high": 0.1},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    gym.make(TUNABLE_MOUNTAIN_CAR_V0_ID, **kwargs)

    def test_factory_appends_task_id_observation(self) -> None:
        env = make_mountaincar_env(task_id=3.5, append_task_id=True)
        try:
            self.assertEqual(env.observation_space.shape, (3,))
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (3,))
            self.assertAlmostEqual(float(obs[-1]), 3.5)
            np.testing.assert_allclose(
                env.observation_space.low,
                np.array([-1.2, -0.07, 3.5], dtype=np.float32),
            )
            np.testing.assert_allclose(
                env.observation_space.high,
                np.array([0.6, 0.07, 3.5], dtype=np.float32),
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
