"""Tests for safe-RL helpers, learning curves, and GIF writing."""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from projects.safe_policy_optimisation.utils.learning_curves import (
    CallableUnshieldedRewardCurveCallback,
    LearningCurveLogger,
    evaluate_unshielded_total_rewards,
)
from projects.safe_policy_optimisation.utils.safe_rl import (
    EpisodeMetrics,
    aggregate_training_violations,
    aggregate_violations,
    build_safe_rl_baseline,
    evaluate_policy,
    make_minipacman_cost_fn,
    make_minipacman_env,
    make_safe_rl_env,
    minipacman_state_cost,
    save_gif,
    training_episode_rows,
)
from projects.safe_policy_optimisation.tests.helpers import (
    AlwaysOnePolicy,
    AlwaysStayPolicy,
    TwoStepEnv,
)

class MiniPacmanCostTests(unittest.TestCase):
    def test_state_cost_uses_masa_labels(self) -> None:
        env = make_minipacman_env(max_episode_steps=5)
        try:
            safe_obs, _ = env.reset(seed=0)
            ghost_obs = next(
                state
                for state in range(env.observation_space.n)
                if "ghost" in env.unwrapped.label_fn(state)
            )

            self.assertEqual(minipacman_state_cost(env, safe_obs), 0.0)
            self.assertEqual(minipacman_state_cost(env, ghost_obs), 1.0)
        finally:
            env.close()

    def test_cost_callback_reads_next_state(self) -> None:
        env = make_minipacman_env(max_episode_steps=5)
        try:
            cost_fn = make_minipacman_cost_fn(env)
            ghost_obs = next(
                state
                for state in range(env.observation_space.n)
                if "ghost" in env.unwrapped.label_fn(state)
            )

            self.assertEqual(cost_fn(0, 0, 0.0, ghost_obs, False, False, {}), 1.0)
        finally:
            env.close()

class ViolationAggregationTests(unittest.TestCase):
    def test_aggregate_violations_reports_count_and_percentage(self) -> None:
        episodes = [
            EpisodeMetrics(episode=0, reward=1.0, cost=0.0, length=4, violated=False),
            EpisodeMetrics(episode=1, reward=0.0, cost=1.0, length=5, violated=True),
            EpisodeMetrics(episode=2, reward=0.0, cost=2.0, length=6, violated=True),
            EpisodeMetrics(episode=3, reward=1.0, cost=0.0, length=7, violated=False),
        ]

        summary = aggregate_violations(episodes)

        self.assertEqual(summary["episodes"], 4.0)
        self.assertEqual(summary["violation_count"], 2.0)
        self.assertEqual(summary["violation_percentage"], 50.0)
        self.assertAlmostEqual(summary["mean_episode_cost"], 0.75)

    def test_aggregate_training_violations_uses_exploration_records(self) -> None:
        records = [
            {"episode": 0, "end_timestep": 4, "reward": 1.0, "cost": 0.0, "length": 4, "violated": False},
            {"episode": 1, "end_timestep": 9, "reward": 0.0, "cost": 2.0, "length": 5, "violated": True},
        ]

        summary = aggregate_training_violations(records)
        rows = training_episode_rows("ppo_lagrangian", records)

        self.assertEqual(summary["training_episode_count"], 2)
        self.assertEqual(summary["training_violation_count"], 1)
        self.assertEqual(summary["training_violation_percentage"], 50.0)
        self.assertEqual(rows[0]["algorithm"], "ppo_lagrangian")

    def test_evaluate_policy_returns_episode_records(self) -> None:
        env = make_minipacman_env(max_episode_steps=3)
        try:
            episodes = evaluate_policy(
                AlwaysStayPolicy(),
                env,
                cost_limit=0.0,
                episodes=2,
                seed=123,
            )

            self.assertEqual(len(episodes), 2)
            self.assertEqual([episode.episode for episode in episodes], [0, 1])
            self.assertTrue(all(episode.length <= 3 for episode in episodes))
            self.assertTrue(all(isinstance(episode.violated, bool) for episode in episodes))
        finally:
            env.close()

class LearningCurveLoggerTests(unittest.TestCase):
    def test_unshielded_reward_evaluation_audits_greedy_actions_against_shield(self) -> None:
        rows = evaluate_unshielded_total_rewards(
            AlwaysOnePolicy(),
            lambda: TwoStepEnv(),
            episodes=1,
            seed=0,
            reward_threshold=0.0,
            shield_mask=np.array([[1, 0], [0, 1]], dtype=int),
        )

        self.assertEqual(rows[0]["total_reward"], 2.0)
        self.assertEqual(rows[0]["proposed_action_checks"], 2)
        self.assertEqual(rows[0]["unsafe_proposed_action_count"], 1)
        self.assertEqual(rows[0]["unsafe_proposed_action_rate"], 0.5)
        self.assertEqual(rows[0]["shield_alignment_rate"], 0.5)

    def test_learning_curve_logger_writes_tensorboard_and_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logger = LearningCurveLogger(
                curve_dir=root / "curves",
                tensorboard_log_dir=root / "tb",
            )
            try:
                logger.log_exploration_unsafe(timestep=1, unsafe_this_step=2, checked_this_step=5)
                logger.log_exploration_unsafe(timestep=2, unsafe_this_step=1, checked_this_step=5)
                summary = logger.log_unshielded_evaluation(
                    timestep=2,
                    episode_rows=[
                        {
                            "episode": 0,
                            "total_reward": 4.0,
                            "length": 3,
                            "success": True,
                            "proposed_action_checks": 3,
                            "unsafe_proposed_action_count": 1,
                            "unsafe_proposed_action_rate": 1 / 3,
                            "shield_alignment_rate": 2 / 3,
                        },
                        {
                            "episode": 1,
                            "total_reward": 2.0,
                            "length": 5,
                            "success": False,
                            "proposed_action_checks": 5,
                            "unsafe_proposed_action_count": 1,
                            "unsafe_proposed_action_rate": 0.2,
                            "shield_alignment_rate": 0.8,
                        },
                    ],
                )
            finally:
                logger.close()

            with (root / "curves" / "exploration_unsafe_actions.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                exploration_rows = list(csv.DictReader(handle))
            with (root / "curves" / "evaluation_unshielded_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                summary_rows = list(csv.DictReader(handle))
            with (root / "curves" / "evaluation_unshielded_episodes.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                episode_rows = list(csv.DictReader(handle))

            self.assertEqual(exploration_rows[-1]["cumulative_unsafe"], "3")
            self.assertEqual(summary["mean_total_reward"], 3.0)
            self.assertEqual(summary["proposed_action_checks"], 8)
            self.assertEqual(summary["unsafe_proposed_action_count"], 2)
            self.assertEqual(summary["unsafe_proposed_action_rate"], 0.25)
            self.assertEqual(summary["shield_alignment_rate"], 0.75)
            self.assertEqual(summary_rows[0]["mean_total_reward"], "3.0")
            self.assertEqual(summary_rows[0]["unsafe_proposed_action_count"], "2")
            self.assertEqual(len(episode_rows), 2)
            self.assertEqual(episode_rows[0]["proposed_action_checks"], "3")
            self.assertTrue(list((root / "tb").glob("events.out.tfevents.*")))

    def test_callable_reward_curve_records_final_timestep_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            logger = LearningCurveLogger(
                curve_dir=root / "curves",
                tensorboard_log_dir=root / "tb",
            )
            model = AlwaysOnePolicy()
            model.num_timesteps = 3
            callback = CallableUnshieldedRewardCurveCallback(
                env_factory=lambda: TwoStepEnv(),
                curve_logger=logger,
                eval_freq=10,
                eval_episodes=1,
                seed=0,
                reward_threshold=0.0,
                shield_mask=np.array([[1, 0], [0, 1]], dtype=int),
            )
            try:
                callback.record_final_evaluation(model)
                callback.record_final_evaluation(model)
            finally:
                logger.close()

            with (root / "curves" / "evaluation_unshielded_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                summary_rows = list(csv.DictReader(handle))

            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["timestep"], "3")
            self.assertEqual(callback.evaluations[0]["timestep"], 3)

class GifWritingTests(unittest.TestCase):
    def test_save_gif_writes_file(self) -> None:
        frames = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 255, dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episode.gif"
            save_gif(path, frames, fps=2.0)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
