"""Exact transition budgets, runtime shielding, and seed-level uncertainty."""

from __future__ import annotations

import sys
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_inference_time_comparison as benchmark
from provably_safe_policy_optimisation import Shield


class ToyPolicy:
    def predict(self, observation, deterministic):
        assert deterministic
        return np.array(0), None


class ToyEnv:
    def __init__(self):
        self.steps = 0
        self.resets = []
        self.actions = []

    def reset(self, seed):
        self.resets.append(seed)
        return 0, {}

    def step(self, action):
        self.steps += 1
        self.actions.append(action)
        return 0, 12345, False, self.steps % 3 == 0, {}


class InferenceTimingTests(unittest.TestCase):
    def test_exact_steps_resets_and_progress(self):
        env = ToyEnv()
        progress = []
        result = benchmark.measure_rollout(
            ToyPolicy(),
            env,
            None,
            steps=10,
            seed=10000,
            progress=progress.append,
            interval=4,
        )
        self.assertEqual(env.steps, 10)
        self.assertEqual(env.resets, [10000, 10001, 10002, 10003])
        self.assertEqual([row["steps_completed"] for row in progress], [4, 8, 10])
        self.assertEqual(result["shield_s"], 0)
        self.assertGreater(result["rollout_s"], result["inference_s"])
        self.assertNotIn("reward", result)

    def test_shield_on_overrides_unsafe_actions(self):
        env = ToyEnv()
        shield = Shield([[0, 1]], seed=0)
        result = benchmark.measure_rollout(ToyPolicy(), env, shield, steps=7, seed=0)
        self.assertEqual(env.actions, [1] * 7)
        self.assertEqual(shield.diagnostics()["checked"], 7)
        self.assertGreater(result["shield_s"], 0)

    def test_sample_standard_error(self):
        mean, error = benchmark.mean_two_se([1, 2, 3])
        self.assertEqual(mean, 2)
        self.assertAlmostEqual(error, 2 / np.sqrt(3))

    def test_all_120_checkpoints_exist(self):
        manifest = benchmark.build_manifest(
            Path("/tmp/inference-manifest-test"), list(range(10)), None, 1000000, 1000
        )
        self.assertEqual(len(manifest["jobs"]), 120)
        self.assertEqual(len({job["id"] for job in manifest["jobs"]}), 120)
        self.assertTrue(all(job["steps"] == 1000000 for job in manifest["jobs"]))

    def test_segment_manifest_uses_region_first_checkpoints(self):
        manifest = benchmark.build_manifest(
            Path("/tmp/inference-segment-manifest-test"), list(range(10)), None,
            1000000, 1000, pspo_variant="segment",
        )
        self.assertEqual(len(manifest["jobs"]), 120)
        for job in manifest["jobs"]:
            if job["method"] == "pspo":
                self.assertIn("/segment_lid/two_hidden/", job["model_path"])
                self.assertFalse(job["config"]["adaptive"]["verify_first"])
                self.assertEqual(job["config"]["adaptive"]["safe_region_shape"], "segment")

    def test_table_uses_paired_reductions_and_one_standard_error(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"output_dir": directory, "pspo_variant": "segment",
                        "jobs": [{"environment": "toy", "environment_label": "Toy"}]}
            pspo = [float(seed + 1) for seed in range(10)]
            shield = [2 * value + 3 for value in pspo]
            groups = {( "toy", method): [
                {"seed": seed, "environment_steps": 1000000, "inference_s": value}
                for seed, value in enumerate(values)
            ] for method, values in (("pspo", pspo), ("ppo_shield", shield))}
            benchmark.write_latency_table(manifest, groups)
            with (Path(directory) / "latency_summary.csv").open() as handle:
                row = next(csv.DictReader(handle))
            percentages = [100 * (s - p) / s for p, s in zip(pspo, shield)]
            self.assertAlmostEqual(float(row["reduction_s_mean"]), np.mean(np.subtract(shield, pspo)))
            self.assertAlmostEqual(float(row["reduction_percent_mean"]), np.mean(percentages))
            self.assertAlmostEqual(float(row["reduction_percent_se"]), np.std(percentages, ddof=1) / np.sqrt(10))
            self.assertAlmostEqual(float(row["pspo_inference_s_se"]), np.std(pspo, ddof=1) / np.sqrt(10))
            table = (Path(directory) / "latency_table.tex").read_text()
            self.assertIn("PSPO-LS", table)
            self.assertIn("standard error over 10 paired seeds", table)

    def test_table_requires_all_ten_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"output_dir": directory, "pspo_variant": "segment",
                        "jobs": [{"environment": "toy", "environment_label": "Toy"}]}
            benchmark.write_latency_table(manifest, {})
            self.assertFalse((Path(directory) / "latency_table.tex").exists())


if __name__ == "__main__":
    unittest.main()
