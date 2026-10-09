"""Matched shaping, native baseline checkpoints and raw final evaluation."""

from __future__ import annotations

import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

from projects.safe_policy_optimisation.scripts import (
    run_frozenlake_safe_baseline_shaping as worker,
)
from projects.safe_policy_optimisation.scripts import (
    run_frozenlake_shaping_sweep as sweep,
)
from projects.safe_policy_optimisation.tests.test_frozenlake_shaping_sweep_unittest import (
    settings,
)
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (
    SparseFrozenLake,
)
from projects.safe_policy_optimisation.utils.frozen_lake_reward_shaping import (
    FrozenLakePotentialReward,
)
from projects.safe_policy_optimisation.utils.safe_rl import load_checkpoint_model


class FrozenLakeSafeBaselineShapingTests(unittest.TestCase):
    def test_four_baselines_have_forty_paired_jobs_and_matching_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            args = settings(Path(temp) / "new")
            args.methods = list(sweep.SAFE_BASELINES)
            args.size = 128
            args.total_timesteps = 400000
            manifest = sweep.build_manifest(args, list(range(32, 72)))
            self.assertEqual(len(manifest["jobs"]), 40)
            self.assertEqual(len({j["cpu"] for j in manifest["jobs"]}), 40)
            self.assertEqual(manifest["methods"], list(sweep.SAFE_BASELINES))
            for job in manifest["jobs"]:
                flags = job["command"][job["command"].index("-u") + 2 :]
                if job["method"] == "ppo_shield":
                    flags.remove("--worker")
                    parsed = sweep.ppo.build_parser().parse_args(flags)
                else:
                    parsed = worker.build_parser().parse_args(flags)
                    self.assertEqual(parsed.algorithm, job["method"])
                    self.assertEqual(parsed.cost_limit, 0)
                    self.assertEqual(parsed.cost_gamma, 0.99)
                self.assertEqual(parsed.size, 128)
                self.assertEqual(parsed.seed, job["seed"])
                self.assertEqual(parsed.total_timesteps, 400000)
                self.assertEqual(parsed.n_steps, 2048)
                self.assertEqual(parsed.batch_size, 64)
                self.assertEqual(parsed.n_epochs, 10)
                self.assertEqual(parsed.gamma, 0.999)
                self.assertEqual(parsed.max_episode_steps, 5000)
                self.assertEqual(parsed.shaping_scale, 1)
                self.assertEqual(parsed.timeout_mode, "bootstrap")
                self.assertEqual(parsed.eval_episodes, 100)
                self.assertEqual(parsed.curve_eval_freq, 20000)
            for method in sweep.SAFE_BASELINES:
                self.assertEqual(
                    [j["seed"] for j in manifest["jobs"] if j["method"] == method],
                    list(range(10)),
                )
                self.assertEqual(
                    manifest["runtime_evaluation_shield"][method],
                    method == "ppo_shield",
                )
            paths = manifest["source_sha256"]
            self.assertIn("core/safe_rl_baselines/ppo_lagrangian.py", paths)
            self.assertIn("core/safe_rl_baselines/cpo.py", paths)

    def test_step_audit_preserves_costs_and_timeout_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            for action, timeout in ((0, True), (1, False)):
                raw = gym.wrappers.TimeLimit(
                    SparseFrozenLake(desc=["SF", "HG"], is_slippery=False),
                    max_episode_steps=1,
                )
                logger = worker.ppo.TrainingRewardLogger(
                    Path(temp) / f"episode{action}.csv",
                    gamma=0.999,
                    scale=1,
                )
                env = worker.TrainingAuditEnv(
                    FrozenLakePotentialReward(raw, gamma=0.999),
                    logger,
                )
                env.reset(seed=0)
                _, _, terminated, truncated, info = env.step(action)
                self.assertEqual(info["cost"], int(not timeout))
                self.assertEqual(truncated and not terminated, timeout)
                self.assertEqual(logger.episode, 1)
                self.assertLess(logger.max_telescoping_error, 1e-10)
                logger.close()
                env.close()
                with (Path(temp) / f"episode{action}.csv").open() as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["truncated"], str(timeout))
                self.assertEqual(row["safe_trajectory"], str(timeout))

    def test_all_cost_baselines_smoke_and_checkpoint_reproduce_raw_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            common = [
                "--output-dir",
                temp,
                "--size",
                "16",
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
            for algorithm in worker.ALGORITHM_NAMES:
                args = worker.build_parser().parse_args(
                    common + ["--algorithm", algorithm, "--run-id", algorithm]
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = worker.run(args)
                self.assertEqual(summary["final_timesteps"], 64)
                self.assertLess(summary["max_discounted_telescoping_error"], 1e-10)
                self.assertFalse(summary["metrics"]["reward_shaping_enabled"])
                root = Path(temp) / algorithm
                config = sweep.read_json(root / "config.json")
                self.assertTrue(
                    config["initialization"]["actor_and_critics_unchanged_by_shaping"]
                )
                self.assertFalse(config["initialization"]["warm_start"])
                with (root / "episodes.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    self.assertAlmostEqual(
                        float(row["total_reward"]),
                        int(row["success"] == "True") - 0.001 * int(row["length"]),
                    )
                raw = gym.make(worker.ENV_ID, max_episode_steps=16, size=16)
                model, _ = load_checkpoint_model(root / "model.pt", env=raw)
                try:
                    _, metrics = worker.ppo.evaluate(
                        model,
                        lambda: gym.make(worker.ENV_ID, max_episode_steps=16, size=16),
                        episodes=2,
                        seed=10000,
                    )
                finally:
                    raw.close()
                self.assertAlmostEqual(
                    metrics["reward"]["mean_total_reward"],
                    summary["metrics"]["reward"]["mean_total_reward"],
                )

            control_args = worker.build_parser().parse_args(
                common
                + [
                    "--algorithm",
                    "ppo_lagrangian",
                    "--shaping-scale",
                    "0",
                    "--run-id",
                    "control",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                control = worker.run(control_args)
            shaped = sweep.read_json(Path(temp) / "ppo_lagrangian" / "summary.json")
            self.assertEqual(
                control["initial_policy_sha256"], shaped["initial_policy_sha256"]
            )

    def test_supervisor_accepts_native_checkpoint_and_aggregates_new_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "_logs").mkdir()
            jobs, records = [], []
            for algorithm in worker.ALGORITHM_NAMES:
                directory = root / algorithm / "seed0"
                sweep.atomic_json(
                    directory / "summary.json",
                    {
                        "training_seconds_excluding_curve_evaluation": 10.0,
                    },
                )
                sweep.atomic_json(
                    directory / "metrics.json",
                    {
                        "reward": {"mean_total_reward": 0.25},
                        "safety": {"safety_rate": 1.0},
                        "success": {"success_rate": 0.8},
                    },
                )
                # Use an existing native checkpoint format without disguising it as a ZIP.
                worker.torch.save({"test": np.zeros(1)}, directory / "model.pt")
                job = {
                    "id": f"{algorithm}/seed0",
                    "method": algorithm,
                    "seed": 0,
                    "cpu": 0,
                    "directory": str(directory),
                    "command": [sys.executable, "-c", "pass"],
                }
                jobs.append(job)
                record = sweep.run_job(job, root)
                self.assertEqual(record["status"], "complete")
                records.append(record)
            sweep.write_report(root, {"jobs": jobs}, records)
            aggregate = sweep.read_json(root / "aggregate.json")["methods"]
            self.assertEqual(set(aggregate), set(worker.ALGORITHM_NAMES))
            for method in aggregate.values():
                self.assertEqual(method["seed_count"], 1)
                self.assertIsNone(method["total_reward"]["two_standard_errors"])


if __name__ == "__main__":
    unittest.main()
