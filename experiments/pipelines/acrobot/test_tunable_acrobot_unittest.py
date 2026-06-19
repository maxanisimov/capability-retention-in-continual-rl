"""Unit tests for TunableAcrobotEnv."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import gymnasium as gym

    from experiments.pipelines.acrobot.core.env import (
        TUNABLE_ACROBOT_V1_ID,
        TunableAcrobotEnv,
        make_acrobot_env,
    )
except Exception as exc:  # pragma: no cover - dependency guard
    gym = None  # type: ignore[assignment]
    TUNABLE_ACROBOT_V1_ID = "TunableAcrobot-v1"
    TunableAcrobotEnv = None  # type: ignore[assignment]
    make_acrobot_env = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_ENV_AVAILABLE = bool(gym is not None and TunableAcrobotEnv is not None)


@unittest.skipUnless(
    _ENV_AVAILABLE,
    f"Gymnasium/TunableAcrobot dependencies unavailable: {_IMPORT_ERROR!r}",
)
class TunableAcrobotTests(unittest.TestCase):
    def test_gym_make_accepts_tunable_dynamics(self):
        env = gym.make(
            TUNABLE_ACROBOT_V1_ID,
            gravity=12.0,
            link_length_1=1.2,
            link_mass_2=1.4,
            available_torque=(-2.0, 0.0, 2.0),
            torque_noise_max=0.05,
        )
        try:
            base = env.unwrapped
            self.assertIsInstance(base, TunableAcrobotEnv)
            self.assertEqual(base.gravity, 12.0)
            self.assertEqual(base.LINK_LENGTH_1, 1.2)
            self.assertEqual(base.LINK_MASS_2, 1.4)
            self.assertEqual(base.AVAIL_TORQUE, [-2.0, 0.0, 2.0])
            self.assertEqual(base.torque_noise_max, 0.05)
        finally:
            env.close()

    def test_observation_and_action_spaces_update(self):
        env = gym.make(
            TUNABLE_ACROBOT_V1_ID,
            max_vel_1=5.0,
            max_vel_2=6.0,
            available_torque=(-1.0, -0.5, 0.0, 0.5, 1.0),
        )
        try:
            self.assertEqual(env.action_space.n, 5)
            np.testing.assert_allclose(
                env.unwrapped.observation_space.high,
                np.asarray([1.0, 1.0, 1.0, 1.0, 5.0, 6.0], dtype=np.float32),
            )
        finally:
            env.close()

    def test_reset_bounds_options_and_initial_state(self):
        env = gym.make(TUNABLE_ACROBOT_V1_ID, reset_low=-0.02, reset_high=0.03)
        try:
            for seed in range(5):
                _obs, info = env.reset(seed=seed)
                state = np.asarray(info["initial_state"], dtype=np.float32)
                self.assertTrue(np.all(state >= -0.02))
                self.assertTrue(np.all(state <= 0.03))

            _obs, info = env.reset(seed=123, options={"low": 0.01, "high": 0.02})
            state = np.asarray(info["initial_state"], dtype=np.float32)
            self.assertTrue(np.all(state >= 0.01))
            self.assertTrue(np.all(state <= 0.02))

            env.reset(seed=123, options={"initial_state": (0.1, 0.2, 0.3, 0.4)})
            np.testing.assert_allclose(env.unwrapped.state, np.asarray([0.1, 0.2, 0.3, 0.4]))
        finally:
            env.close()

    def test_goal_reaching_step_reports_success_and_safe(self):
        env = gym.make(TUNABLE_ACROBOT_V1_ID, terminal_height=-2.1)
        try:
            env.reset(seed=0, options={"initial_state": (0.0, 0.0, 0.0, 0.0)})
            _obs, _reward, terminated, _truncated, info = env.step(1)
            self.assertTrue(terminated)
            self.assertIs(info["is_success"], True)
            self.assertIs(info["safe"], True)
        finally:
            env.close()

    def test_invalid_settings_raise(self):
        invalid_kwargs = [
            {"gravity": float("inf")},
            {"link_length_1": 0.0},
            {"link_length_2": -1.0},
            {"link_mass_1": 0.0},
            {"link_mass_2": -1.0},
            {"link_com_pos_1": -0.1},
            {"link_moi": 0.0},
            {"max_vel_1": 0.0},
            {"max_vel_2": -1.0},
            {"available_torque": ()},
            {"torque_noise_max": -0.1},
            {"dt": 0.0},
            {"book_or_nips": "paper"},
            {"reset_low": 0.1, "reset_high": -0.1},
            {"initial_state": (0.0, 0.0, 0.0)},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    env = gym.make(TUNABLE_ACROBOT_V1_ID, **kwargs)
                    env.close()

    def test_factory_appends_task_id(self):
        env = make_acrobot_env(task_id=math.e, append_task_id=True)
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (7,))
            self.assertAlmostEqual(float(obs[-1]), math.e, places=6)
            self.assertEqual(env.observation_space.shape, (7,))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
