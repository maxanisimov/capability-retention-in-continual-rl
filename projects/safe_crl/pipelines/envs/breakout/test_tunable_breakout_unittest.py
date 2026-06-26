"""Unit tests for TunableBreakoutEnv."""

from __future__ import annotations

import unittest

try:
    import gymnasium as gym

    from projects.safe_crl.pipelines.envs.breakout.core.env import (
        TUNABLE_ALE_BREAKOUT_V5_ID,
        TUNABLE_BREAKOUT_V5_ID,
        TunableBreakoutEnv,
        make_breakout_env,
    )
except Exception as exc:  # pragma: no cover - dependency guard
    gym = None  # type: ignore[assignment]
    TUNABLE_ALE_BREAKOUT_V5_ID = "TunableALE/Breakout-v5"
    TUNABLE_BREAKOUT_V5_ID = "TunableBreakout-v5"
    TunableBreakoutEnv = None  # type: ignore[assignment]
    make_breakout_env = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_ENV_AVAILABLE = bool(gym is not None and TunableBreakoutEnv is not None)


@unittest.skipUnless(
    _ENV_AVAILABLE,
    f"Gymnasium/TunableBreakout dependencies unavailable: {_IMPORT_ERROR!r}",
)
class TunableBreakoutTests(unittest.TestCase):
    def _make_or_skip(self, *args, **kwargs):
        try:
            return gym.make(*args, **kwargs)
        except Exception as exc:
            self.skipTest(f"ALE Breakout ROM/environment unavailable: {exc!r}")

    def test_gym_make_accepts_tunable_ale_settings(self):
        env = self._make_or_skip(
            TUNABLE_BREAKOUT_V5_ID,
            obs_type="ram",
            frameskip=[2, 5],
            repeat_action_probability=0.0,
            full_action_space=True,
            max_num_frames_per_episode=1000,
        )
        try:
            base = env.unwrapped
            self.assertIsInstance(base, TunableBreakoutEnv)
            self.assertEqual(base.game, "breakout")
            self.assertEqual(base.obs_type, "ram")
            self.assertEqual(base.frameskip, (2, 5))
            self.assertEqual(base.repeat_action_probability, 0.0)
            self.assertIs(base.full_action_space, True)
            self.assertEqual(base.max_num_frames_per_episode, 1000)
            self.assertEqual(env.observation_space.shape, (128,))
        finally:
            env.close()

    def test_namespaced_alias_is_registered(self):
        env = self._make_or_skip(
            TUNABLE_ALE_BREAKOUT_V5_ID,
            obs_type="ram",
            frameskip=1,
        )
        try:
            self.assertEqual(env.unwrapped.game, "breakout")
            self.assertEqual(env.unwrapped.frameskip, 1)
        finally:
            env.close()

    def test_step_adds_common_info_fields(self):
        env = self._make_or_skip(
            TUNABLE_BREAKOUT_V5_ID,
            frameskip=1,
            repeat_action_probability=0.0,
        )
        try:
            env.reset(seed=0)
            _obs, _reward, _terminated, _truncated, info = env.step(0)
            self.assertIn("safe", info)
            self.assertIn("is_success", info)
            self.assertIs(info["safe"], True)
        finally:
            env.close()

    def test_invalid_settings_raise_before_rom_load(self):
        invalid_kwargs = [
            {"obs_type": "pixels"},
            {"frameskip": 0},
            {"frameskip": (0, 4)},
            {"frameskip": (5, 2)},
            {"repeat_action_probability": -0.1},
            {"repeat_action_probability": 1.1},
            {"continuous_action_threshold": -0.1},
            {"max_num_frames_per_episode": 0},
            {"mode": -1},
            {"difficulty": -1},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TunableBreakoutEnv(**kwargs)

    def test_factory_creates_configured_env(self):
        env = self._make_or_skip(
            TUNABLE_BREAKOUT_V5_ID,
            obs_type="ram",
            frameskip=1,
        )
        env.close()

        env = make_breakout_env(obs_type="ram", frameskip=1)
        try:
            self.assertEqual(env.unwrapped.obs_type, "ram")
            self.assertEqual(env.unwrapped.frameskip, 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
