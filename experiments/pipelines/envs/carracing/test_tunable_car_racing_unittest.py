"""Unit tests for TunableCarRacingEnv."""

from __future__ import annotations

import unittest

try:
    import gymnasium as gym

    from experiments.pipelines.envs.carracing.core.env import (
        TUNABLE_CAR_RACING_V3_ID,
        TunableCarRacingEnv,
        make_carracing_env,
    )
except Exception as exc:  # pragma: no cover - dependency guard
    gym = None  # type: ignore[assignment]
    TUNABLE_CAR_RACING_V3_ID = "TunableCarRacing-v3"
    TunableCarRacingEnv = None  # type: ignore[assignment]
    make_carracing_env = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_ENV_AVAILABLE = bool(gym is not None and TunableCarRacingEnv is not None)


@unittest.skipUnless(
    _ENV_AVAILABLE,
    f"Gymnasium/TunableCarRacing dependencies unavailable: {_IMPORT_ERROR!r}",
)
class TunableCarRacingTests(unittest.TestCase):
    def test_gym_make_accepts_tunable_settings(self):
        env = gym.make(
            TUNABLE_CAR_RACING_V3_ID,
            continuous=False,
            lap_complete_percent=0.85,
            domain_randomize=True,
            discrete_steer=0.7,
            discrete_gas=0.3,
            discrete_brake=0.9,
            frame_dt=0.04,
            per_step_penalty=0.2,
            off_track_penalty=-50.0,
            playfield=250.0,
        )
        try:
            base = env.unwrapped
            self.assertIsInstance(base, TunableCarRacingEnv)
            self.assertFalse(base.continuous)
            self.assertEqual(env.action_space.n, 5)
            self.assertEqual(base.lap_complete_percent, 0.85)
            self.assertIs(base.domain_randomize, True)
            self.assertEqual(base.discrete_steer, 0.7)
            self.assertEqual(base.discrete_gas, 0.3)
            self.assertEqual(base.discrete_brake, 0.9)
            self.assertEqual(base.frame_dt, 0.04)
            self.assertEqual(base.per_step_penalty, 0.2)
            self.assertEqual(base.off_track_penalty, -50.0)
            self.assertEqual(base.playfield, 250.0)
        finally:
            env.close()

    def test_reset_and_step_use_penalty_and_info_fields(self):
        env = gym.make(
            TUNABLE_CAR_RACING_V3_ID,
            continuous=False,
            per_step_penalty=100.0,
        )
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (96, 96, 3))
            _obs, reward, _terminated, _truncated, info = env.step(0)
            self.assertLessEqual(reward, 0.0)
            self.assertIn("is_success", info)
            self.assertIn("safe", info)
            self.assertIs(info["safe"], True)
        finally:
            env.close()

    def test_invalid_settings_raise(self):
        invalid_kwargs = [
            {"discrete_steer": -0.1},
            {"discrete_gas": -0.1},
            {"discrete_brake": -0.1},
            {"frame_dt": 0.0},
            {"world_velocity_iterations": 0},
            {"world_position_iterations": -1},
            {"per_step_penalty": -0.1},
            {"off_track_penalty": float("nan")},
            {"playfield": 0.0},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    env = gym.make(TUNABLE_CAR_RACING_V3_ID, **kwargs)
                    env.close()

    def test_factory_creates_configured_env(self):
        env = make_carracing_env(continuous=False, discrete_gas=0.4)
        try:
            self.assertFalse(env.unwrapped.continuous)
            self.assertEqual(env.unwrapped.discrete_gas, 0.4)
            self.assertEqual(env.action_space.n, 5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
