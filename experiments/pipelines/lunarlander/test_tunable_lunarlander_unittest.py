"""Unit tests for TunableLunarLander `info['is_success']` contract."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.envs.box2d.lunar_lander import LunarLander
except Exception:  # pragma: no cover - dependency guard
    gym = None  # type: ignore[assignment]
    LunarLander = None  # type: ignore[assignment]

try:
    from experiments.pipelines.lunarlander.core.env.tunable_lunarlander import (
        TunableLunarLander,
    )
    from experiments.pipelines.lunarlander.core.env.wrappers import (
        LunarLanderCrashSafetyWrapper,
    )
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - dependency guard
    TunableLunarLander = None  # type: ignore[assignment]
    LunarLanderCrashSafetyWrapper = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


_ENV_AVAILABLE = bool(
    gym is not None
    and LunarLander is not None
    and TunableLunarLander is not None
    and LunarLanderCrashSafetyWrapper is not None,
)


@unittest.skipUnless(
    _ENV_AVAILABLE,
    f"LunarLander/Gymnasium dependencies unavailable: {_IMPORT_ERROR!r}",
)
class TunableLunarLanderSuccessFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TunableLunarLander(render_mode=None, continuous=False)

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_info_contains_is_success_false(self) -> None:
        _, info = self.env.reset(seed=0)
        self.assertIn("is_success", info)
        self.assertIs(info["is_success"], False)

    def test_non_terminal_step_sets_is_success_false(self) -> None:
        fake_obs = np.zeros((8,), dtype=np.float32)
        with patch.object(
            LunarLander,
            "step",
            return_value=(fake_obs, 1.0, False, False, {}),
        ):
            _, _, terminated, truncated, info = self.env.step(0)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("is_success", info)
        self.assertIs(info["is_success"], False)

    def test_terminal_failure_step_sets_is_success_false(self) -> None:
        fake_obs = np.zeros((8,), dtype=np.float32)
        with patch.object(
            LunarLander,
            "step",
            return_value=(fake_obs, -100.0, True, False, {}),
        ):
            _, reward, terminated, truncated, info = self.env.step(0)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.isclose(float(reward), -100.0))
        self.assertIn("is_success", info)
        self.assertIs(info["is_success"], False)

    def test_terminal_api_success_step_sets_is_success_true(self) -> None:
        fake_obs = np.zeros((8,), dtype=np.float32)
        with patch.object(
            LunarLander,
            "step",
            return_value=(fake_obs, 100.0, True, False, {}),
        ):
            _, reward, terminated, truncated, info = self.env.step(0)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.isclose(float(reward), 100.0))
        self.assertIn("is_success", info)
        self.assertIs(info["is_success"], True)

    def test_wrapper_preserves_is_success_and_adds_safe(self) -> None:
        class _DummySuccessEnv(gym.Env):  # type: ignore[misc]
            metadata = {}

            def __init__(self) -> None:
                super().__init__()
                self.action_space = gym.spaces.Discrete(2)
                self.observation_space = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(8,),
                    dtype=np.float32,
                )
                self.game_over = False

            def reset(self, *, seed: int | None = None, options: dict | None = None):
                super().reset(seed=seed)
                return np.zeros((8,), dtype=np.float32), {"is_success": False}

            def step(self, action):
                obs = np.zeros((8,), dtype=np.float32)
                return obs, 100.0, True, False, {"is_success": True}

        wrapped_env = LunarLanderCrashSafetyWrapper(_DummySuccessEnv())
        try:
            wrapped_env.reset(seed=0)
            _, _, _, _, info = wrapped_env.step(0)
        finally:
            wrapped_env.close()

        self.assertIn("is_success", info)
        self.assertIn("safe", info)
        self.assertIs(info["is_success"], True)
        self.assertIs(info["safe"], True)


if __name__ == "__main__":
    unittest.main()
