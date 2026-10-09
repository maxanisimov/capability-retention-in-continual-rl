"""Shaping sweeps pair budgets/seeds and distinguish nominal/deployed shielding."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

from projects.safe_policy_optimisation.scripts import (
    run_frozenlake_shaping_sweep as sweep,
)
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (
    SparseFrozenLake,
    synthesise_shield,
)


def settings(root: Path, *, smoke: bool = False):
    return argparse.Namespace(
        output_dir=root,
        size=32,
        total_timesteps=200000,
        eval_episodes=100,
        screen_session="test-frozenlake-shaping",
        smoke=smoke,
    )


class FrozenLakeShapingSweepTests(unittest.TestCase):
    def test_manifest_has_thirty_distinct_paired_jobs_and_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            manifest = sweep.build_manifest(settings(root), list(range(32, 62)))
            jobs = manifest["jobs"]
            self.assertEqual(len(jobs), 30)
            self.assertEqual(len({j["cpu"] for j in jobs}), 30)
            self.assertEqual(len({j["directory"] for j in jobs}), 30)
            self.assertTrue((root / "source_snapshot.zip").exists())
            for method in sweep.METHODS:
                self.assertEqual(
                    [j["seed"] for j in jobs if j["method"] == method], list(range(10))
                )
            for job in jobs:
                command = job["command"]
                args = command[command.index("-u") + 2 :]
                if job["method"] == "pspo":
                    parsed = sweep.pspo.build_parser().parse_args(args)
                    self.assertEqual(parsed.safety_frequency, 100)
                    self.assertEqual(parsed.lid_iters, 200)
                    self.assertEqual(parsed.state_representation, "one_hot")
                else:
                    parsed = sweep.ppo.build_parser().parse_args(
                        [x for x in args if x != "--worker"]
                    )
                self.assertEqual(parsed.size, 32)
                self.assertEqual(parsed.total_timesteps, 200000)
                self.assertEqual(parsed.eval_episodes, 100)
                self.assertEqual(parsed.max_episode_steps, 1250)
                self.assertEqual(parsed.shaping_scale, 1)
                self.assertEqual(parsed.gamma, 0.999)
                self.assertEqual(parsed.seed, job["seed"])
                self.assertEqual(parsed.n_steps, 2048)
                self.assertEqual(parsed.n_epochs, 10)
            self.assertEqual(
                manifest["runtime_evaluation_shield"],
                {"pspo": False, "ppo_shield": True, "ppo": False},
            )
            with self.assertRaises(FileExistsError):
                sweep.build_manifest(settings(root), list(range(32, 62)))

    def test_insufficient_or_duplicate_cpus_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            for cpus in (list(range(29)), [32] * 30):
                with self.assertRaises(ValueError):
                    sweep.build_manifest(settings(root), cpus)
            self.assertFalse(root.exists())

    def test_selected_ppo_pspo_128_sweep_has_twenty_jobs_and_400k_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            args = settings(root)
            args.methods = ["ppo", "pspo"]
            args.size = 128
            args.total_timesteps = 400000
            manifest = sweep.build_manifest(args, list(range(32, 52)))
            self.assertEqual(manifest["methods"], ["ppo", "pspo"])
            self.assertEqual(manifest["requested_timesteps_per_run"], 400000)
            self.assertEqual(len(manifest["jobs"]), 20)
            self.assertEqual(len({j["cpu"] for j in manifest["jobs"]}), 20)
            self.assertEqual(
                manifest["runtime_evaluation_shield"], {"ppo": False, "pspo": False}
            )
            for method in args.methods:
                self.assertEqual(
                    [j["seed"] for j in manifest["jobs"] if j["method"] == method],
                    list(range(10)),
                )
            for job in manifest["jobs"]:
                command = job["command"]
                self.assertNotIn("--worker", command)
                parser = (
                    sweep.pspo.build_parser()
                    if job["method"] == "pspo"
                    else sweep.ppo.build_parser()
                )
                parsed = parser.parse_args(command[command.index("-u") + 2 :])
                self.assertEqual(parsed.size, 128)
                self.assertEqual(parsed.total_timesteps, 400000)
                self.assertEqual(parsed.max_episode_steps, 5000)
                self.assertEqual(parsed.eval_episodes, 100)
                self.assertEqual(parsed.shaping_scale, 1)
                self.assertEqual(parsed.gamma, 0.999)
                self.assertEqual(parsed.seed, job["seed"])
                self.assertEqual(parsed.n_steps, 2048)
                self.assertEqual(
                    ((parsed.total_timesteps + parsed.n_steps - 1) // parsed.n_steps)
                    * parsed.n_steps,
                    401408,
                )

    def test_invalid_method_selection_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            args = settings(root)
            for methods in ([], ["ppo", "ppo"], ["unknown"]):
                args.methods = methods
                with self.assertRaises(ValueError):
                    sweep.build_manifest(args, list(range(32, 62)))
            self.assertFalse(root.exists())

    def test_shielded_evaluation_corrects_unsafe_actions_without_shaping(self):
        def factory():
            return gym.wrappers.TimeLimit(
                SparseFrozenLake(desc=["SFG", "FHF"], is_slippery=False),
                max_episode_steps=3,
            )

        raw = factory()
        mask, _, _, _ = synthesise_shield(raw.unwrapped)
        raw.close()

        class UnsafePolicy:
            def predict(self, obs, deterministic=True):
                return np.array(2 if int(obs) == 0 else 1), None

        for shield_on in (True, False):
            rows, metrics = sweep.evaluate_shield_model(
                UnsafePolicy(),
                factory,
                mask=mask,
                episodes=4,
                seed=0,
                shield_on=shield_on,
            )
            self.assertFalse(metrics["reward_shaping_enabled"])
            self.assertEqual(metrics["runtime_shield"], shield_on)
            self.assertEqual(metrics["safety"]["safety_rate"], float(shield_on))
            for row in rows:
                self.assertAlmostEqual(
                    row["total_reward"], int(row["success"]) - 0.001 * row["length"]
                )

    def test_shield_worker_pairs_random_initialization_with_plain_ppo(self):
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
            args = sweep.ppo.build_parser().parse_args(common + ["--run-id", "shield"])
            plain = sweep.ppo.build_parser().parse_args(common + ["--run-id", "plain"])
            with contextlib.redirect_stdout(io.StringIO()):
                shielded = sweep.run_shield_worker(args)
                control = sweep.ppo.run(plain)
            self.assertEqual(
                shielded["initial_policy_sha256"], control["initial_policy_sha256"]
            )
            self.assertEqual(shielded["final_timesteps"], 64)
            self.assertEqual(shielded["training_shield_diagnostics"]["checked"], 64)
            self.assertLess(shielded["max_discounted_telescoping_error"], 1e-10)
            root = Path(temp) / "shield"
            config = sweep.read_json(root / "config.json")
            self.assertTrue(
                config["initialization"]["actor_and_critic_unchanged_by_shaping"]
            )
            self.assertFalse(config["initialization"]["warm_start"])
            self.assertTrue(shielded["metrics"]["runtime_shield"])
            self.assertFalse(shielded["nominal_metrics"]["runtime_shield"])
            for name in ("episodes.csv", "episodes_nominal.csv"):
                with (root / name).open() as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    self.assertAlmostEqual(
                        float(row["total_reward"]),
                        int(row["success"] == "True") - 0.001 * int(row["length"]),
                    )

    def test_aggregate_uses_training_seed_sample_se_and_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = []
            completed = []
            for seed, reward in enumerate((1.0, 3.0)):
                directory = root / "ppo" / f"seed{seed}"
                sweep.atomic_json(
                    directory / "summary.json",
                    {"training_seconds_excluding_curve_evaluation": 10.0 + seed},
                )
                sweep.atomic_json(
                    directory / "metrics.json",
                    {
                        "reward": {"mean_total_reward": reward},
                        "safety": {"safety_rate": 1.0},
                        "success": {"success_rate": 1.0},
                    },
                )
                job = {
                    "id": f"ppo/seed{seed}",
                    "method": "ppo",
                    "seed": seed,
                    "directory": str(directory),
                }
                jobs.append(job)
                completed.append({"job_id": job["id"], "status": "complete"})
            jobs.append(
                {
                    "id": "ppo/seed2",
                    "method": "ppo",
                    "seed": 2,
                    "directory": str(root / "ppo/seed2"),
                }
            )
            sweep.write_report(root, {"jobs": jobs}, completed)
            aggregate = sweep.read_json(root / "aggregate.json")["methods"]["ppo"]
            self.assertEqual(aggregate["seed_count"], 2)
            self.assertEqual(aggregate["total_reward"]["mean"], 2.0)
            self.assertAlmostEqual(
                aggregate["total_reward"]["two_standard_errors"], 2.0
            )
            status = sweep.read_json(root / "status.json")
            self.assertEqual(status["complete"], 2)
            self.assertEqual(status["running_or_starting"], 1)


if __name__ == "__main__":
    unittest.main()
