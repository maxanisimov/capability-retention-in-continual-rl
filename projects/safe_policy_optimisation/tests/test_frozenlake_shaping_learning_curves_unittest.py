"""Learning curves use seed means, raw rewards and genuinely paired projection data."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from projects.safe_policy_optimisation.scripts import (
    plot_frozenlake_shaping_learning_curves as curves,
)


class FrozenLakeLearningCurveTests(unittest.TestCase):
    def test_two_sample_standard_errors_and_constant_sanity_check(self):
        self.assertEqual(
            curves.mean_two_se([1, 3]),
            {
                "mean": 2.0,
                "two_standard_errors": 2.0,
                "seed_count": 2,
            },
        )
        self.assertEqual(curves.mean_two_se([1] * 10)["two_standard_errors"], 0)
        self.assertIsNone(curves.mean_two_se([1])["two_standard_errors"])
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                curves.mean_two_se(values)

    def test_completed_episode_bins_boundaries_and_no_zero_filled_gaps(self):
        rows = [
            {
                "end_timestep": step,
                "raw_reward": reward,
                "shaped_reward": reward - 1,
                "goal_reached": "True",
                "safe_trajectory": "True",
            }
            for step, reward in ((20, 1), (21, 3), (39, 5), (61, 7))
        ]
        binned = curves.bin_training(rows, [0, 20, 40, 60, 64])
        self.assertEqual([r["timestep"] for r in binned], [10, 30, 62])
        self.assertEqual([r["reward"] for r in binned], [1, 4, 7])
        self.assertEqual(binned[1]["episodes"], 2)
        self.assertEqual(binned[1]["shaped_reward"], 3)

    def test_aggregate_seed_means_not_pooled_episodes(self):
        points = [
            {
                "method": "ppo",
                "channel": "exploration",
                "timestep": 10,
                "seed": seed,
                "episodes": count,
                "reward": value,
                "goal": 1.0,
                "safety": 1.0,
            }
            for seed, count, value in ((0, 100, 1.0), (1, 1, 3.0))
        ]
        row = curves.aggregate_points(points)[0]
        self.assertEqual(row["seed_count"], 2)
        self.assertEqual(row["reward_mean"], 2.0)
        self.assertEqual(row["reward_two_se"], 2.0)
        with self.assertRaises(ValueError):
            curves.aggregate_points(points + [points[0]])

    def test_final_ten_episode_curve_not_substituted_for_hundred_episode_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "status.json").write_text(json.dumps({"complete": 20, "failed": 0}))
            for method in ("ppo", "pspo"):
                for seed in range(10):
                    directory = root / method / f"seed{seed}"
                    (directory / "learning_curves").mkdir(parents=True)
                    config = {
                        "requested_timesteps": 400000,
                        "env_kwargs": {"size": 128, "step_penalty": 0.001},
                    }
                    summary = {
                        "final_timesteps": 401408,
                        "adaptive_diagnostics": {
                            "safety_update_events": [
                                {"timestep": 204800, "decision": "projected"},
                                {"timestep": 401408, "decision": "projected"},
                            ]
                        },
                    }
                    for filename, obj in (
                        ("config.json", config),
                        ("summary.json", summary),
                    ):
                        (directory / filename).write_text(json.dumps(obj))
                    metrics = {
                        "reward_shaping_enabled": False,
                        "evaluation_policy": "greedy_unshielded",
                        "success": {"success_rate": 1.0},
                        "safety": {"safety_rate": 1.0},
                    }
                    for filename, reward, n in (
                        ("initial_metrics.json", -5, 10),
                        ("metrics.json", 0.2, 100),
                        ("pre_finalization_metrics.json", 0.7, 10),
                    ):
                        (directory / filename).write_text(
                            json.dumps(
                                {
                                    **metrics,
                                    "eval_episodes": n,
                                    "reward": {"mean_total_reward": reward},
                                }
                            )
                        )
                    curves.write_csv(
                        directory / "training_episodes.csv",
                        [
                            {
                                "end_timestep": 5000,
                                "length": 5000,
                                "raw_reward": -5,
                                "shaped_reward": -6,
                                "goal_reached": "False",
                                "safe_trajectory": "True",
                            }
                        ],
                    )
                    curves.write_csv(
                        directory / "learning_curves/evaluation_unshielded_summary.csv",
                        [
                            {
                                "timestep": t,
                                "episodes": 10,
                                "mean_total_reward": 0.6,
                                "success_rate": 1,
                                "safety_rate": 1,
                            }
                            for t in (300000, 320000, 340000, 360000, 380000, 401408)
                        ],
                    )
                    curves.write_csv(
                        directory / "episodes.csv",
                        [
                            {
                                "episode": i,
                                "total_reward": 0.5 if i < 10 else 1 / 6,
                                "success": "True",
                                "safe_trajectory": "True",
                            }
                            for i in range(100)
                        ],
                    )
            points, details = curves.load_data(root)
            aggregate = curves.aggregate_points(points)
            final = curves.select(aggregate, "pspo", "final_100_episodes")[0]
            self.assertAlmostEqual(final["reward_mean"], 0.2)
            proposal = curves.select(aggregate, "pspo", "evaluation")[-1]
            self.assertAlmostEqual(proposal["reward_mean"], 0.7)
            paired = details["pspo"]["final_projection_paired"]
            self.assertAlmostEqual(paired["after"]["reward"]["mean"], 0.5)
            self.assertAlmostEqual(paired["change"]["reward"]["mean"], -0.2)


if __name__ == "__main__":
    unittest.main()
