"""Cadence propagation tests for the source-preserving sweep adapter."""

import unittest

from projects.safe_policy_optimisation.scripts.run_frozenlake_segment_frequency_sweep import (
    set_frequency,
)


class SegmentFrequencyTests(unittest.TestCase):
    def test_all_ten_worker_commands_and_metadata(self):
        manifest = {
            "jobs": [
                {
                    "method": "pspo",
                    "seed": seed,
                    "command": [
                        "python",
                        "--safety-frequency",
                        "100",
                        "--verify-first",
                    ],
                }
                for seed in range(10)
            ],
            "pspo_safety_frequency_rollouts": 100,
        }
        result = set_frequency(manifest, 1)
        self.assertEqual(result["pspo_safety_frequency_rollouts"], 1)
        for job in result["jobs"]:
            self.assertEqual(job["command"][2], "1")
            self.assertIn("--verify-first", job["command"])

    def test_nonpositive_cadence_rejected(self):
        for frequency in (0, -1):
            with self.assertRaises(ValueError):
                set_frequency({"jobs": []}, frequency)

    def test_missing_or_duplicate_cadence_rejected(self):
        for command in (
            ["python"],
            ["--safety-frequency", "1", "--safety-frequency", "100"],
        ):
            with self.assertRaises(ValueError):
                set_frequency({"jobs": [{"method": "pspo", "command": command}]}, 1)


if __name__ == "__main__":
    unittest.main()
