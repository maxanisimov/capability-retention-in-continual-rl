"""Exact transition budgets, runtime shielding, and seed-level uncertainty."""

from __future__ import annotations

import sys
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


if __name__ == "__main__":
    unittest.main()
