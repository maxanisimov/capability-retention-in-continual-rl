"""Tests for raw-reward and across-seed efficiency plotting semantics."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/plot_frozenlake_ppo_pspo_reward_efficiency.py"
)
SPEC = importlib.util.spec_from_file_location("reward_efficiency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RewardEfficiencyTests(unittest.TestCase):
    def test_raw_reward_bins_and_right_closed_boundaries(self):
        rows = [
            dict(
                end_timestep=t,
                raw_reward=r,
                shaped_reward=999,
                goal_reached="true",
                safe_trajectory="true",
            )
            for t, r in [(10, 0.9), (20, 0.7), (21, -0.1), (40, 0.5), (41, 500)]
        ]
        points = MODULE.bin_training(rows, [0, 20, 40])
        self.assertAlmostEqual(points[0]["reward"], 0.8)
        self.assertAlmostEqual(points[1]["reward"], 0.2)
        self.assertEqual([p["timestep"] for p in points], [20, 40])

    def test_empty_bin_not_zero_filled(self):
        with self.assertRaises(ValueError):
            MODULE.bin_training([], [0, 20])

    def test_two_sample_standard_errors(self):
        stats = MODULE.mean_two_se([1, 3])
        self.assertEqual(stats["mean"], 2)
        self.assertAlmostEqual(stats["two_se"], 2)
        self.assertEqual(MODULE.mean_two_se([1] * 10)["two_se"], 0)

    def test_rollout_timing_prefix_and_final_anchor(self):
        log = "\n".join(
            f"| time_elapsed | {seconds} |\n| total_timesteps | {step} |"
            for seconds, step in [(2, 2048), (5, 4096), (10, 6144)]
        )
        steps, seconds = MODULE.rollout_times(log, 5000, 7)
        self.assertEqual(steps.tolist(), [0, 2048, 4096, 5000])
        self.assertEqual(seconds.tolist(), [0, 2, 5, 7])

    def test_ten_distinct_seeds_and_separate_final_channel(self):
        points = [
            dict(
                size=16,
                method="pspo",
                channel=channel,
                timestep=204800,
                seed=seed,
                reward=reward,
                goal=1,
                safety=1,
                elapsed_seconds=5,
            )
            for channel, reward in [("evaluation", 0.5), ("final", 0.9)]
            for seed in range(10)
        ]
        rows = MODULE.aggregate(points)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["reward_mean"] for r in rows], [0.5, 0.9])
        with self.assertRaises(ValueError):
            MODULE.aggregate(points[:-1])
        points[-1]["seed"] = 0
        with self.assertRaises(ValueError):
            MODULE.aggregate(points)


if __name__ == "__main__":
    unittest.main()
