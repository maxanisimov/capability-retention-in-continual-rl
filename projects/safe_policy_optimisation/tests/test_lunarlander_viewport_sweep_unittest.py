"""Viewport shield semantics, full-box coverage, final audits, and sweep setup."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from continuous_state_shields import (
    LunarLanderViewportShield,
    LunarLanderViewportShieldConfig,
)
from continuous_state_shields.lunar_lander_viewport import viewport_certificate_arrays

from projects.safe_policy_optimisation.scripts import (
    run_lunarlander_viewport_sweep as sweep,
)
from projects.safe_policy_optimisation.stages.train_ppo_shield import (
    make_continuous_state_shield,
)
from projects.safe_policy_optimisation.stages.train_pspo_continuous import (
    LunarLanderViewportCostWrapper,
    evaluate_viewport_safety,
    lunarlander_out_of_view,
    parse_args,
    validate_lunarlander_certificate,
)


class FinalEscapeEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (8,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)
    lander = None
    game_over = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        observation = np.zeros(8, dtype=np.float32)
        observation[0] = 1.1
        return observation, -100.0, True, False, {}


class LunarLanderViewportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = gym.make("LunarLander-v3", continuous=False)
        self.addCleanup(self.env.close)
        self.low, self.high = self.env.observation_space.low, self.env.observation_space.high

    def test_masks_at_all_six_thresholds_and_endpoints(self) -> None:
        for m in sweep.MS:
            shield = LunarLanderViewportShield(LunarLanderViewportShieldConfig(m=m))
            for x, expected in ((m, [1]), (1, [1]), (-m, [3]), (-1, [3]),
                                (0, [0, 1, 2, 3]), (m - 0.001, [0, 1, 2, 3]),
                                (-m + 0.001, [0, 1, 2, 3]), (1.01, [0, 1, 2, 3])):
                obs = np.zeros(8)
                obs[0] = x
                self.assertEqual(shield.get_safe_actions(obs), expected)
                obs[2], obs[4] = -9, 6  # Position-only; velocity/angle don't change rule.
                self.assertEqual(shield.get_safe_actions(obs), expected)

    def test_config_rejects_invalid_thresholds_or_swapped_actions(self) -> None:
        for m in (0, 1, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LunarLanderViewportShieldConfig(m=m)
        with self.assertRaises(ValueError):
            LunarLanderViewportShieldConfig(push_left_action=3)

    def test_complete_boxes_cover_bands_and_all_other_coordinates(self) -> None:
        for m in sweep.MS:
            low, high, masks = viewport_certificate_arrays(m, self.low, self.high)
            dataset = torch.utils.data.TensorDataset(*(torch.as_tensor(array) for array in (low, high, masks)))
            shield = LunarLanderViewportShield(LunarLanderViewportShieldConfig(m=m))
            validate_lunarlander_certificate(dataset, shield)
            self.assertEqual(masks.tolist(), [[False, False, False, True], [False, True, False, False]])
            self.assertLessEqual(float(low[1, 0]), m)
            self.assertGreaterEqual(float(high[0, 0]), -m)
            np.testing.assert_array_equal(low[:, 1:], np.tile(self.low[1:], (2, 1)))
            np.testing.assert_array_equal(high[:, 1:], np.tile(self.high[1:], (2, 1)))
            self.assertEqual(float(low[0, 0]), -1)
            self.assertEqual(float(high[1, 0]), 1)

    def test_incomplete_certificate_or_wrong_mask_rejected(self) -> None:
        low, high, masks = viewport_certificate_arrays(0.8, self.low, self.high)
        shield = LunarLanderViewportShield()
        for arrays in ((low[:1], high[:1], masks[:1]), (low, high, np.ones((2, 4), dtype=bool))):
            with self.assertRaisesRegex(ValueError, "both complete edge boxes"):
                validate_lunarlander_certificate(torch.utils.data.TensorDataset(*(torch.as_tensor(array) for array in arrays)), shield)

    def test_registered_config_and_stage_cli(self) -> None:
        shield = make_continuous_state_shield("lunarlander-viewport", "LunarLander-v3", '{"m": 0.75}')
        self.assertEqual(shield.config.m, 0.75)
        with self.assertRaises(ValueError):
            make_continuous_state_shield("lunarlander-viewport", "MountainCar-v0")
        args = parse_args(["--base-policy-path", "unused.pt", "--env-id", "LunarLander-v3",
                           "--continuous-shield", "lunarlander-viewport", "--mountaincar-shaped-reward", "false"])
        self.assertEqual(args.max_episode_steps, 1000)

    def test_upright_side_engine_impulse_directions(self) -> None:
        velocities = {}
        for action in (0, 1, 3):
            self.env.reset(seed=123)
            raw = self.env.unwrapped
            for body in (raw.lander, *raw.legs):
                body.linearVelocity = (0, 0)
                body.angularVelocity = 0
            raw.lander.angle = 0
            obs, *_ = self.env.step(action)
            velocities[action] = float(obs[2])
        self.assertLess(velocities[1], velocities[0])
        self.assertGreater(velocities[3], velocities[0])

    def test_cost_and_audit_include_final_escape(self) -> None:
        env = LunarLanderViewportCostWrapper(FinalEscapeEnv())
        self.assertEqual(env.reset()[1]["cost"], 0)
        self.assertEqual(env.step(0)[-1]["cost"], 1)
        model = SimpleNamespace(predict=lambda obs, deterministic: (np.array(0), None))
        summary = evaluate_viewport_safety(model, FinalEscapeEnv, LunarLanderViewportShield(),
                                           apply_shield=False, episodes=2, seed=10000)
        self.assertEqual(summary["out_of_view_episodes"], 2)
        self.assertEqual(summary["safe_trajectory_rate"], 0)
        self.assertEqual(summary["no_truncation_rate"], 1)
        self.assertEqual(summary["successful_landing_rate"], 0)
        self.assertTrue(summary["per_episode"][0]["terminated"])
        self.assertFalse(summary["per_episode"][0]["truncated"])

    def test_cost_uses_raw_coordinate_not_rounded_observation(self) -> None:
        self.env.reset(seed=0)
        self.env.unwrapped.lander.position = (21, 10)
        self.assertTrue(lunarlander_out_of_view(self.env, np.zeros(8)))
        self.env.unwrapped.lander.position = (10, 10)
        observation = np.zeros(8)
        observation[0] = 1.1
        self.assertFalse(lunarlander_out_of_view(self.env, observation))


class LunarLanderViewportSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "sweep"
        self.record = sweep.prepare(self.root, False)

    def test_sixty_unique_pairs_fixed_budget_and_separate_initial_policies(self) -> None:
        self.assertEqual(len(self.record["pairs"]), 60)
        self.assertEqual(len({pair["id"] for pair in self.record["pairs"]}), 60)
        for pair in self.record["pairs"]:
            commands = sweep.build_commands(self.root, pair, self.record["settings"])
            args = parse_args(commands[1][3:])
            self.assertEqual(args.total_timesteps, 500000)
            self.assertEqual(args.rashomon_n_iters, 200)
            self.assertEqual(args.rashomon_objective, "weighted_width")
            self.assertEqual(args.adaptive_granularity, "train_phase")
            self.assertEqual(args.evaluation_policy, "unshielded")
            self.assertEqual(args.early_stop_eval_freq, 0)
            self.assertIn(pair["id"], str(args.base_policy_path))
            self.assertEqual(args.env_id, "LunarLander-v3")
            self.assertEqual(args.ent_coef, 0.01)
        sweep.scheduler.verify_provenance(self.root, self.record)

    def test_reports_pending_and_preserves_existing_root(self) -> None:
        self.assertEqual(sweep.read_json(self.root / "aggregate.json")["completed_pairs"], 0)
        self.assertIn("No truncation includes crashes", (self.root / "report.md").read_text())
        with self.assertRaises(FileExistsError):
            sweep.prepare(self.root, False)

    def test_failed_initialisation_never_starts_rl(self) -> None:
        pair = self.record["pairs"][0]
        process = SimpleNamespace(pid=123, wait=lambda: 1)
        with patch.object(sweep.os, "sched_getaffinity", return_value={8}), \
                patch.object(sweep.subprocess, "Popen", return_value=process) as launch:
            with self.assertRaisesRegex(RuntimeError, "initialisation exited"):
                sweep.worker(self.root, pair["id"])
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(sweep.read_json(self.root / pair["id"] / "status.json")["status"], "failed")

    def test_partial_aggregation_counts_seeds_and_computes_two_se(self) -> None:
        results = {}
        for pair in self.record["pairs"]:
            if pair["seed"] > 1:
                continue
            run = self.root / pair["id"]
            run.mkdir(parents=True)
            result = {**pair, "mean_total_reward": -200 + 20 * pair["seed"],
                      "safety_rate": 0.5 + 0.5 * pair["seed"], "action_compliance": 1.0,
                      "no_truncation_rate": 1.0, "successful_landing_rate": 0.2,
                      "reward_threshold_success_rate": 0.1}
            results[pair["id"]] = result
            sweep.write_json(run / "result.json", result)
            sweep.write_json(run / "status.json", {"status": "complete"})
        with patch.object(sweep, "validate_result", side_effect=lambda run, pair, config: results[pair["id"]]):
            aggregate = sweep.write_report(self.root)
        self.assertEqual(aggregate["completed_pairs"], 12)
        self.assertEqual(aggregate["requested_pairs"], 60)
        self.assertEqual(aggregate["failed_pairs"], [])
        for interval in aggregate["intervals"]:
            self.assertEqual(interval["completed_seeds"], 2)
            self.assertEqual(interval["mean_total_reward"], -190)
            self.assertAlmostEqual(interval["mean_total_reward_2se"], 20)
            self.assertEqual(interval["safety_rate"], 0.75)
            self.assertEqual(interval["action_compliance"], 1)
        self.assertIn("75.00 ± 50.00", (self.root / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
