"""Plain-PPO FrozenLake launches preserve budgets, random init and goal metrics."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.safe_policy_optimisation.scripts import (
    run_frozenlake_ppo_sweep as sweep,
)
from projects.safe_policy_optimisation.scripts import (
    run_stochastic_frozenlake_baselines as baseline,
)
from projects.safe_policy_optimisation.stages import train_ppo


def settings(size: int) -> dict:
    return {
        "size": size,
        "max_episode_steps": 5000 * size // 128,
        "total_timesteps": 2_000_000,
        "evaluation_episodes": 100,
        "step_penalty": 0.001,
        "success_rate": 0.8,
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.999,
    }


class FrozenLakePlainPPOTests(unittest.TestCase):
    def test_plain_ppo_stage_flags_match_reference_without_warm_start(self):
        self.assertIn("ppo", baseline.METHODS)
        self.assertEqual(baseline.DEFAULT_METHODS, ("ppo_shield", "cpo"))
        for size in (128, 256):
            argv = baseline.training_arguments(
                "ppo",
                Path("/tmp/reference"),
                settings(size),
                7,
                Path("/tmp/output/ppo/seed7"),
                smoke=False,
            )
            args = train_ppo.build_parser().parse_args(argv)
            self.assertEqual(args.total_timesteps, 2_000_000)
            self.assertEqual(args.n_steps, 2048)
            self.assertEqual(args.n_epochs, 10)
            self.assertEqual(args.max_episode_steps, 5000 * size // 128)
            self.assertEqual(args.eval_episodes, 100)
            self.assertEqual(args.seed, 7)
            self.assertEqual(args.n_hidden, 2)
            self.assertEqual(args.hidden_dim, 64)
            self.assertIsNone(args.init_policy_path)
            self.assertFalse(args.mountaincar_shaped_reward)
            self.assertNotIn("--evaluation-policy", argv)
            self.assertNotIn("--jobs", argv)

    def test_smoke_is_small_and_separate(self):
        argv = baseline.training_arguments(
            "ppo",
            Path("/tmp/reference"),
            settings(256),
            0,
            Path("/tmp/output/_smoke/ppo/seed0"),
            smoke=True,
        )
        args = train_ppo.build_parser().parse_args(argv)
        self.assertEqual(args.total_timesteps, 64)
        self.assertEqual(args.n_steps, 64)
        self.assertEqual(args.n_epochs, 1)
        self.assertEqual(args.max_episode_steps, 128)
        self.assertEqual(args.eval_episodes, 2)
        self.assertEqual(args.curve_eval_freq, 0)

    def test_dispatch_uses_plain_ppo_not_shielded_ppo(self):
        argv = baseline.training_arguments(
            "ppo",
            Path("/tmp/reference"),
            settings(128),
            0,
            Path("/tmp/output/ppo/seed0"),
            smoke=True,
        )
        with patch.object(train_ppo, "run", return_value={"plain": True}) as run:
            self.assertEqual(baseline.run_stage("ppo", argv), {"plain": True})
            run.assert_called_once()
            self.assertIsNone(run.call_args.args[0].init_policy_path)

    def test_final_evaluation_converts_numpy_scalar_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            args = train_ppo.build_parser().parse_args(
                [
                    "--env-id",
                    baseline.ENV_ID,
                    "--env-kwargs",
                    '{"size":16}',
                    "--total-timesteps",
                    "8",
                    "--n-steps",
                    "8",
                    "--batch-size",
                    "8",
                    "--n-epochs",
                    "1",
                    "--n-hidden",
                    "0",
                    "--max-episode-steps",
                    "8",
                    "--eval-episodes",
                    "2",
                    "--curve-eval-freq",
                    "0",
                    "--device",
                    "cpu",
                    "--output-dir",
                    directory,
                    "--run-id",
                    "test",
                ]
            )
            with patch.object(
                train_ppo.PPO, "predict", return_value=(np.array(0), None)
            ):
                train_ppo.run(args)
            metrics = baseline.read_json(Path(directory) / "test/metrics.json")
            self.assertEqual(metrics["eval_episodes"], 2)

    def test_goal_metrics_include_negative_return_goal_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {
                "reward": {"mean_total_reward": -2.5},
                "safety": {"safety_rate": 1.0},
                "success": {"success_mode": "reward_threshold", "success_rate": 0.0},
            }
            baseline.write_json(root / "metrics.json", original)
            baseline.write_json(
                root / "summary.json", {"training": {"training_safety_rate": 1.0}}
            )
            with (root / "episodes.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["reward", "length"])
                writer.writeheader()
                writer.writerows(
                    [{"reward": -1, "length": 2000}, {"reward": -5, "length": 5000}]
                )
            baseline.normalise_plain_ppo_goal_metrics(root, 0.001)
            actual = baseline.read_json(root / "metrics.json")
            self.assertEqual(actual["success"]["success_rate"], 0.5)
            self.assertEqual(actual["success"]["success_count"], 1)
            self.assertEqual(actual["success"]["success_mode"], "goal_reached")
            self.assertEqual(actual["reward"], original["reward"])
            self.assertEqual(actual["safety"], original["safety"])
            self.assertEqual(
                baseline.read_json(root / "metrics_reward_threshold.json"), original
            )
            row = baseline.seed_row("ppo", root, 0.001)
            self.assertEqual(row["success_rate"], 0.5)
            self.assertEqual(row["safety_rate"], 1.0)

    def test_manifest_contains_twenty_unique_jobs_and_cpu_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            sources = {size: Path(directory) / f"source{size}" for size in (128, 256)}
            records = {
                sources[size]: {"settings": settings(size), "input_sha256": {}}
                for size in (128, 256)
            }
            with patch.object(
                baseline, "verify_inputs", side_effect=lambda path: records[path]
            ):
                manifest = sweep.build_manifest(
                    output, list(range(8, 28)), sources=sources
                )
                with self.assertRaises(FileExistsError):
                    sweep.build_manifest(output, list(range(8, 28)), sources=sources)
            jobs = manifest["jobs"]
            self.assertEqual(len(jobs), 20)
            self.assertEqual(len({job["cpu"] for job in jobs}), 20)
            self.assertEqual(len({job["directory"] for job in jobs}), 20)
            for size in (128, 256):
                self.assertEqual(
                    [job["seed"] for job in jobs if job["size"] == size],
                    list(range(10)),
                )
            for job in jobs:
                self.assertIn("--worker", job["command"])
                self.assertEqual(
                    job["command"][job["command"].index("--method") + 1], "ppo"
                )
                self.assertNotIn("--init-policy-path", job["stage_arguments"])
                self.assertNotIn("--smoke", job["command"])
            self.assertTrue((output / "source_snapshot.zip").is_file())

    def test_manifest_rejects_insufficient_or_duplicate_cpus_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            for cpus in (list(range(8, 27)), [8] * 20):
                with self.assertRaises(ValueError):
                    sweep.build_manifest(output, cpus)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
