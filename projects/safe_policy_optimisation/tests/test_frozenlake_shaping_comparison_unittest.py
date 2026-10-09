"""Final reports use paired seed aggregates and reject protocol mismatches."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from projects.safe_policy_optimisation.scripts import (
    report_frozenlake_shaping_comparison as report,
)


class FrozenLakeShapingComparisonTests(unittest.TestCase):
    def test_seed_sample_two_se_and_constant_pspo_sanity_check(self):
        rows = []
        for method in report.METHOD_LABELS:
            for seed in range(2):
                value = 1.0 if method == "pspo" else 1.0 + 2 * seed
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        **{key: value for key in (*report.KEYS, "lid_seconds")},
                    }
                )
        aggregate = report.aggregate_rows(rows)
        for method, result in aggregate.items():
            self.assertEqual(result["seed_count"], 2)
            self.assertAlmostEqual(
                result["total_reward"]["mean"], 1 if method == "pspo" else 2
            )
            self.assertAlmostEqual(
                result["total_reward"]["two_standard_errors"],
                0 if method == "pspo" else 2,
            )
        with self.assertRaises(ValueError):
            report.aggregate_rows(rows + [rows[0]])

    def test_protocol_accepts_extra_architecture_metadata_but_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            control_dir = root / "ppo/seed0"
            safe_dir = root / "pspo/seed0"
            control_dir.mkdir(parents=True)
            safe_dir.mkdir(parents=True)
            config = {
                "env_kwargs": {"size": 128},
                "max_episode_steps": 5000,
                "requested_timesteps": 400000,
                "architecture": {"hidden_dim": 64, "n_hidden": 2},
                "training_hyperparameters": {"gamma": 0.999, "n_steps": 2048},
                "shaping": {
                    "enabled": True,
                    "scale": 1,
                    "gamma": 0.999,
                    "timeout_mode": "bootstrap",
                    "training_only": True,
                },
            }
            (control_dir / "config.json").write_text(json.dumps(config))
            safe_config = {
                **config,
                "architecture": {
                    **config["architecture"],
                    "input_dim": 16384,
                    "n_actions": 4,
                },
            }
            (safe_dir / "config.json").write_text(json.dumps(safe_config))
            (safe_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "final_timesteps": 401408,
                        "training_seconds_excluding_curve_evaluation": 10.0,
                    }
                )
            )
            metrics = {
                "eval_episodes": 100,
                "reward_shaping_enabled": False,
                "evaluation_policy": "greedy_unshielded",
                "reward": {"mean_total_reward": 0.2},
                "safety": {"safety_rate": 1.0},
                "success": {"success_rate": 1.0},
            }
            (safe_dir / "metrics.json").write_text(json.dumps(metrics))
            rows = [
                {
                    "method": "pspo",
                    "seed": 0,
                    "total_reward": 0.2,
                    "safety_rate": 1.0,
                    "goal_success": 1.0,
                    "training_seconds": 10.0,
                }
            ]
            report.verify_final_protocol(root, root, rows)
            safe_config["requested_timesteps"] = 200000
            (safe_dir / "config.json").write_text(json.dumps(safe_config))
            with self.assertRaises(ValueError):
                report.verify_final_protocol(root, root, rows)
            safe_config["requested_timesteps"] = 400000
            (safe_dir / "config.json").write_text(json.dumps(safe_config))
            metrics["reward_shaping_enabled"] = True
            (safe_dir / "metrics.json").write_text(json.dumps(metrics))
            with self.assertRaises(ValueError):
                report.verify_final_protocol(root, root, rows)


if __name__ == "__main__":
    unittest.main()
