"""Unit tests for PPO utility evaluation and early-stop metric criteria."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

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
    ppo_train,
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
    @staticmethod
    def _make_small_ppo_cfg(*, update_epochs: int = 2) -> PPOConfig:
        return PPOConfig(
            seed=123,
            total_timesteps=8,
            eval_episodes=2,
            rollout_steps=2,
            update_epochs=update_epochs,
            minibatch_size=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            lr=1e-3,
            max_grad_norm=0.5,
            device="cpu",
        )

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

    def test_ppo_train_default_shape_and_no_eval_records_when_tracking_disabled(self) -> None:
        env = _SingleStepEvalEnv(
            rewards=[1.0, 3.0],
            safe_flags=[True, False],
            success_flags=[False, True],
        )
        cfg = self._make_small_ppo_cfg(update_epochs=3)

        result_default = ppo_train(env=env, cfg=cfg, track_eval_params=False)
        self.assertIsInstance(result_default, tuple)
        self.assertEqual(len(result_default), 2)

        env_with_records = _SingleStepEvalEnv(
            rewards=[1.0, 3.0],
            safe_flags=[True, False],
            success_flags=[False, True],
        )
        actor, critic, eval_records = ppo_train(
            env=env_with_records,
            cfg=cfg,
            track_eval_params=False,
            return_eval_checkpoint_records=True,
        )
        self.assertIsInstance(actor, torch.nn.Sequential)
        self.assertIsInstance(critic, torch.nn.Sequential)
        self.assertEqual(eval_records, [])

    def test_ppo_train_eval_checkpoint_records_memory_mode(self) -> None:
        env = _SingleStepEvalEnv(
            rewards=[1.0, 3.0],
            safe_flags=[True, False],
            success_flags=[False, True],
        )
        cfg = self._make_small_ppo_cfg(update_epochs=4)

        actor, critic, eval_records = ppo_train(
            env=env,
            cfg=cfg,
            track_eval_params=True,
            return_eval_checkpoint_records=True,
        )

        del critic
        self.assertEqual(len(eval_records), 2)  # periodic + final (no pre-update early-stop eval)
        self.assertEqual([rec["eval_phase"] for rec in eval_records], ["periodic", "final"])

        for rec in eval_records:
            self.assertIn("timestep", rec)
            self.assertIn("update_idx", rec)
            self.assertIn("metrics", rec)
            self.assertIn("params", rec)
            self.assertNotIn("params_path", rec)
            metrics = rec["metrics"]
            self.assertAlmostEqual(metrics["mean_reward"], 2.0, places=7)
            self.assertAlmostEqual(metrics["std_reward"], 1.0, places=7)
            self.assertAlmostEqual(metrics["failure_rate"], 0.5, places=7)
            self.assertAlmostEqual(metrics["success_rate"], 0.5, places=7)
            params = rec["params"]
            self.assertIsInstance(params, dict)
            self.assertTrue(len(params) > 0)
            for tensor in params.values():
                self.assertIsInstance(tensor, torch.Tensor)
                self.assertEqual(tensor.device.type, "cpu")
                self.assertFalse(tensor.requires_grad)

        first_snapshot = eval_records[0]["params"]
        first_key = next(iter(first_snapshot.keys()))
        snapshot_before = first_snapshot[first_key].clone()
        with torch.no_grad():
            next(actor.parameters()).add_(1.0)
        self.assertTrue(torch.equal(first_snapshot[first_key], snapshot_before))
        self.assertFalse(torch.equal(first_snapshot[first_key], actor.state_dict()[first_key].detach().cpu()))

    def test_ppo_train_eval_checkpoint_records_disk_mode(self) -> None:
        env = _SingleStepEvalEnv(
            rewards=[1.0, 3.0],
            safe_flags=[True, False],
            success_flags=[False, True],
        )
        cfg = self._make_small_ppo_cfg(update_epochs=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            actor, critic, eval_records = ppo_train(
                env=env,
                cfg=cfg,
                track_eval_params=True,
                return_eval_checkpoint_records=True,
                save_eval_params_to_disk=True,
                eval_params_save_dir=tmpdir,
            )

            del actor, critic
            self.assertEqual(len(eval_records), 2)
            saved_paths = []
            for rec in eval_records:
                self.assertIn("params_path", rec)
                self.assertNotIn("params", rec)
                params_path = Path(rec["params_path"])
                saved_paths.append(params_path)
                self.assertTrue(params_path.exists())
                self.assertEqual(params_path.parent, Path(tmpdir))
                self.assertAlmostEqual(rec["metrics"]["mean_reward"], 2.0, places=7)
                self.assertAlmostEqual(rec["metrics"]["std_reward"], 1.0, places=7)
                self.assertAlmostEqual(rec["metrics"]["failure_rate"], 0.5, places=7)
                self.assertAlmostEqual(rec["metrics"]["success_rate"], 0.5, places=7)
                payload = torch.load(params_path, map_location="cpu")
                self.assertIsInstance(payload, dict)
                self.assertTrue(len(payload) > 0)
                self.assertTrue(all(isinstance(v, torch.Tensor) for v in payload.values()))
                self.assertTrue(all(v.device.type == "cpu" for v in payload.values()))

            self.assertEqual(len(saved_paths), len(set(saved_paths)))
            self.assertEqual(len(list(Path(tmpdir).glob("*.pt"))), len(eval_records))


if __name__ == "__main__":
    unittest.main()
