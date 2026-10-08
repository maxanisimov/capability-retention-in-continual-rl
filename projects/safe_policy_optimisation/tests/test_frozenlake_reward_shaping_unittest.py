"""Potential-shaping invariance, episode boundaries, and PPO smoke tests."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from projects.safe_policy_optimisation.scripts.run_frozenlake_ppo_shaping import (
    build_parser,
    run,
)
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (
    SparseFrozenLake,
)
from projects.safe_policy_optimisation.utils.frozen_lake_reward_shaping import (
    FrozenLakePotentialReward,
    goal_distance_potential,
)


class FrozenLakeRewardShapingTests(unittest.TestCase):
    def make_env(
        self, *, desc=None, gamma=0.9, scale=1, cap=None, timeout_mode="bootstrap"
    ):
        raw = SparseFrozenLake(desc=desc or ["SF", "HG"], is_slippery=False)
        if cap is not None:
            raw = gym.wrappers.TimeLimit(raw, max_episode_steps=cap)
        env = FrozenLakePotentialReward(
            raw, gamma=gamma, scale=scale, timeout_mode=timeout_mode
        )
        self.addCleanup(env.close)
        return env

    def test_distance_accounts_for_obstacles_and_normalizes_globally(self):
        grid = np.asarray([list(row) for row in ["SFFF", "HHHF", "GFFF"]])
        distance, potential = goal_distance_potential(grid)
        self.assertEqual(distance[0], 8)  # Manhattan distance would incorrectly be 2.
        self.assertEqual(distance[8], 0)
        self.assertEqual(potential[8], 0)  # Goal must not retain potential one.
        self.assertEqual(potential[0], 0)
        self.assertEqual(potential[7], 0.5)
        self.assertTrue(np.all((potential >= 0) & (potential <= 1)))
        np.testing.assert_equal(potential[grid.reshape(-1) == "H"], 0)

    def test_unreachable_cells_have_zero_potential(self):
        distance, potential = goal_distance_potential(np.asarray([list("SHG")]))
        np.testing.assert_equal(distance, [-1, -1, 0])
        np.testing.assert_equal(potential, [0, 0, 0])

    def test_invalid_maps_are_rejected(self):
        for grid in (np.array([]), np.array([list("SF")]), np.array([list("SGG")])):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                goal_distance_potential(grid)

    def test_progress_signal_and_final_goal_correction(self):
        env = self.make_env()
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(2)
        self.assertEqual(obs, 1)
        self.assertFalse(terminated or truncated)
        self.assertAlmostEqual(reward, -0.001 + 0.9 * 0.5)
        self.assertAlmostEqual(info["reward_shaping"]["raw_reward"], -0.001)
        _, reward, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertAlmostEqual(reward, 0.999 - 0.5)
        self.assertEqual(info["reward_shaping"]["next_potential"], 0)

    def test_hole_correction_preserves_cost_and_success_flags(self):
        env = self.make_env(desc=["SFG", "FHF"])
        env.reset(seed=0)
        env.step(2)
        _, reward, terminated, _, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(info["is_success"])
        self.assertEqual(info["cost"], 1)
        self.assertEqual(info["reward_shaping"]["next_potential"], 0)
        self.assertAlmostEqual(reward, -0.001 - 2 / 3)

    def test_terminated_episode_discounted_return_differs_only_by_start_potential(self):
        for actions in ([2, 2], [2, 0, 2, 2], [2, 1]):
            env = self.make_env(desc=["SFG", "FHF"], scale=0.5)
            env.reset(seed=0)
            initial = env.potential[0]
            shaped = raw = 0
            for t, action in enumerate(actions):
                _, reward, terminated, truncated, info = env.step(action)
                shaped += env.gamma**t * reward
                raw += env.gamma**t * info["reward_shaping"]["raw_reward"]
            self.assertTrue(terminated or truncated)
            self.assertAlmostEqual(shaped - raw, -env.scale * initial, places=12)

    def test_bootstrapped_timeout_retains_next_potential_and_flags(self):
        env = self.make_env(cap=1)
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(2)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reward_shaping"]["next_potential"], 0.5)
        self.assertAlmostEqual(reward, -0.001 + 0.9 * 0.5)
        raw_value = 0.7
        shaped_value = raw_value - env.scale * 0.5
        self.assertAlmostEqual(
            reward + env.gamma * shaped_value,
            -0.001 + env.gamma * raw_value - env.scale * env.potential[0],
        )

    def test_terminal_timeout_correction_disables_sb3_bootstrapping(self):
        vec = DummyVecEnv([lambda: self.make_env(cap=1, timeout_mode="terminal")])
        self.addCleanup(vec.close)
        vec.reset()
        _, rewards, dones, infos = vec.step([2])
        self.assertTrue(dones[0])
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertTrue(infos[0]["reward_shaping"]["original_truncated"])
        self.assertEqual(infos[0]["reward_shaping"]["next_potential"], 0)
        self.assertAlmostEqual(float(rewards[0]), -0.001)

    def test_bootstrap_timeout_is_recognized_by_sb3(self):
        vec = DummyVecEnv([lambda: self.make_env(cap=1)])
        self.addCleanup(vec.close)
        vec.reset()
        _, _, dones, infos = vec.step([2])
        self.assertTrue(dones[0])
        self.assertTrue(infos[0]["TimeLimit.truncated"])
        self.assertIn("terminal_observation", infos[0])

    def test_zero_scale_is_reward_and_dynamics_identical_to_control(self):
        raw = SparseFrozenLake(size=16)
        env = FrozenLakePotentialReward(SparseFrozenLake(size=16), gamma=0.999, scale=0)
        self.addCleanup(raw.close)
        self.addCleanup(env.close)
        raw.reset(seed=17)
        env.reset(seed=17)
        for action in [2, 1, 2, 1, 0, 3] * 10:
            a = raw.step(action)
            b = env.step(action)
            self.assertEqual(a[:4], b[:4])
            info = dict(b[4])
            info.pop("reward_shaping")
            self.assertEqual(a[4], info)
            if a[2] or a[3]:
                raw.reset(seed=17)
                env.reset(seed=17)

    def test_wrapper_does_not_mutate_rewards_or_transitions(self):
        raw = SparseFrozenLake(size=16)
        transitions = copy.deepcopy(raw.P)
        env = FrozenLakePotentialReward(raw, gamma=0.999)
        self.addCleanup(env.close)
        env.reset(seed=0)
        env.step(2)
        self.assertEqual(raw.P, transitions)
        self.assertFalse(env.potential.flags.writeable)

    def test_invalid_parameters_are_rejected(self):
        raw = SparseFrozenLake(size=16)
        self.addCleanup(raw.close)
        for kwargs in (
            {"gamma": 0},
            {"gamma": 1.01},
            {"gamma": float("nan")},
            {"gamma": 0.9, "scale": -1},
            {"gamma": 0.9, "scale": float("inf")},
            {"gamma": 0.9, "timeout_mode": "wrong"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FrozenLakePotentialReward(raw, **kwargs)

    def test_runner_smoke_raw_evaluation_paired_initialization_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            summaries = []
            for scale in (0, 1):
                args = build_parser().parse_args(
                    [
                        "--output-dir",
                        temp,
                        "--run-id",
                        f"scale{scale}",
                        "--shaping-scale",
                        str(scale),
                        "--total-timesteps",
                        "64",
                        "--n-steps",
                        "32",
                        "--batch-size",
                        "16",
                        "--n-epochs",
                        "1",
                        "--max-episode-steps",
                        "16",
                        "--eval-episodes",
                        "2",
                        "--curve-eval-episodes",
                        "2",
                        "--curve-eval-freq",
                        "32",
                        "--verbose",
                        "0",
                    ]
                )
                summaries.append(run(args))
                run_dir = Path(temp) / f"scale{scale}"
                with (run_dir / "training_episodes.csv").open() as f:
                    training = list(csv.DictReader(f))
                self.assertTrue(training)
                self.assertLess(
                    summaries[-1]["max_discounted_telescoping_error"], 1e-10
                )
                with (run_dir / "episodes.csv").open() as f:
                    evaluation = list(csv.DictReader(f))
                for row in evaluation:
                    self.assertAlmostEqual(
                        float(row["total_reward"]),
                        float(row["success"] == "True") - 0.001 * int(row["length"]),
                    )
                metrics = json.loads((run_dir / "metrics.json").read_text())
                self.assertFalse(metrics["reward_shaping_enabled"])
                self.assertEqual(metrics["success"]["success_mode"], "goal_reached")
            self.assertEqual(
                summaries[0]["initial_policy_sha256"],
                summaries[1]["initial_policy_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
