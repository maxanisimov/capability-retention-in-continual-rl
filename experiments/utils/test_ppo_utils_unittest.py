"""Unit tests for PPO utility evaluation and early-stop metric criteria."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch

from experiments.utils.ppo_utils import (
    PPOConfig,
    _early_stop_thresholds_satisfied,
    _is_early_stop_enabled,
    evaluate,
    evaluate_masked,
    evaluate_masked_with_success,
    evaluate_with_success,
)


class _SingleStepEvalEnv(gym.Env):
    """Tiny deterministic env with exactly one step per episode."""

    metadata = {}

    def __init__(
        self,
        *,
        rewards: list[float],
        safe_flags: list[bool],
        success_flags: list[bool],
        include_action_mask: bool = False,
    ) -> None:
        super().__init__()
        if not (len(rewards) == len(safe_flags) == len(success_flags)):
            raise ValueError("rewards/safe_flags/success_flags must have the same length.")
        if len(rewards) == 0:
            raise ValueError("Need at least one scripted episode.")
        self._rewards = rewards
        self._safe_flags = safe_flags
        self._success_flags = success_flags
        self._include_action_mask = include_action_mask
        self._episode_idx = -1
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._episode_idx = (self._episode_idx + 1) % len(self._rewards)
        obs = np.array([0.0], dtype=np.float32)
        info: dict[str, object] = {"is_success": False}
        if self._include_action_mask:
            info["action_mask"] = np.array([1.0, 1.0], dtype=np.float32)
        return obs, info

    def step(self, action: int):
        del action
        obs = np.array([0.0], dtype=np.float32)
        reward = float(self._rewards[self._episode_idx])
        terminated = True
        truncated = False
        info: dict[str, object] = {
            "safe": bool(self._safe_flags[self._episode_idx]),
            "is_success": bool(self._success_flags[self._episode_idx]),
        }
        if self._include_action_mask:
            info["action_mask"] = np.array([1.0, 1.0], dtype=np.float32)
        return obs, reward, terminated, truncated, info


def _make_tiny_actor() -> torch.nn.Sequential:
    actor = torch.nn.Sequential(torch.nn.Linear(1, 2))
    with torch.no_grad():
        layer = actor[0]
        assert isinstance(layer, torch.nn.Linear)
        layer.weight.zero_()
        layer.bias.zero_()
    return actor


class PpoUtilsTests(unittest.TestCase):
    def test_evaluate_with_success_and_backward_compatible_wrapper(self) -> None:
        env = _SingleStepEvalEnv(
            rewards=[1.0, 3.0, 5.0],
            safe_flags=[True, False, True],
            success_flags=[False, True, True],
        )
        actor = _make_tiny_actor()

        mean_r, std_r, failure_rate, success_rate = evaluate_with_success(
            env=env,
            actor=actor,
            episodes=3,
            seed=123,
            device="cpu",
            deterministic=True,
        )
        self.assertAlmostEqual(mean_r, 3.0, places=7)
        self.assertAlmostEqual(std_r, float(np.std([1.0, 3.0, 5.0])), places=7)
        self.assertAlmostEqual(failure_rate, 1.0 / 3.0, places=7)
        self.assertAlmostEqual(success_rate, 2.0 / 3.0, places=7)

        mean_r_old, std_r_old, failure_rate_old = evaluate(
            env=env,
            actor=actor,
            episodes=3,
            seed=123,
            device="cpu",
            deterministic=True,
        )
        self.assertAlmostEqual(mean_r_old, mean_r, places=7)
        self.assertAlmostEqual(std_r_old, std_r, places=7)
        self.assertAlmostEqual(failure_rate_old, failure_rate, places=7)

    def test_evaluate_masked_with_success_and_wrapper(self) -> None:
        env = _SingleStepEvalEnv(
            rewards=[2.0, 2.0, 2.0],
            safe_flags=[True, True, True],
            success_flags=[True, False, True],
            include_action_mask=True,
        )
        actor = _make_tiny_actor()

        mean_r, std_r, failure_rate, success_rate = evaluate_masked_with_success(
            env=env,
            actor=actor,
            episodes=3,
            seed=7,
            device="cpu",
            deterministic=True,
        )
        self.assertAlmostEqual(mean_r, 2.0, places=7)
        self.assertAlmostEqual(std_r, 0.0, places=7)
        self.assertAlmostEqual(failure_rate, 0.0, places=7)
        self.assertAlmostEqual(success_rate, 2.0 / 3.0, places=7)

        mean_r_old, std_r_old, failure_rate_old = evaluate_masked(
            env=env,
            actor=actor,
            episodes=3,
            seed=7,
            device="cpu",
            deterministic=True,
        )
        self.assertAlmostEqual(mean_r_old, mean_r, places=7)
        self.assertAlmostEqual(std_r_old, std_r, places=7)
        self.assertAlmostEqual(failure_rate_old, failure_rate, places=7)

    def test_early_stop_thresholds_include_success_rate(self) -> None:
        cfg = PPOConfig(
            early_stop_reward_threshold=100.0,
            early_stop_failure_rate_threshold=0.1,
            early_stop_success_rate_threshold=0.8,
            early_stop_deterministic_total_reward_threshold=95.0,
        )

        reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
            cfg,
            mean_reward=120.0,
            failure_rate=0.05,
            success_rate=0.9,
        )
        self.assertTrue(reward_ok)
        self.assertTrue(failure_ok)
        self.assertTrue(success_ok)

        reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
            cfg,
            mean_reward=120.0,
            failure_rate=0.05,
            success_rate=0.7,
        )
        self.assertTrue(reward_ok)
        self.assertTrue(failure_ok)
        self.assertFalse(success_ok)

    def test_early_stop_enabled_is_driven_by_thresholds(self) -> None:
        cfg_none = PPOConfig(early_stop=True)
        self.assertFalse(_is_early_stop_enabled(cfg_none))

        cfg_reward = PPOConfig(early_stop=False, early_stop_reward_threshold=123.0)
        self.assertTrue(_is_early_stop_enabled(cfg_reward))

        cfg_failure = PPOConfig(early_stop=False, early_stop_failure_rate_threshold=0.2)
        self.assertTrue(_is_early_stop_enabled(cfg_failure))

        cfg_success = PPOConfig(early_stop=False, early_stop_success_rate_threshold=0.8)
        self.assertTrue(_is_early_stop_enabled(cfg_success))


if __name__ == "__main__":
    unittest.main()
