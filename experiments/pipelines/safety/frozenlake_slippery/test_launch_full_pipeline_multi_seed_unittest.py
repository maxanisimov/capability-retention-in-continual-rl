"""Unit tests for the FrozenLake slippery shield safety full-pipeline launcher."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.safety.frozenlake_slippery.cli import launch_full_pipeline_multi_seed as launcher


class FrozenLakeSafetyFullLauncherTests(unittest.TestCase):
    def test_dry_run_schedules_seed_pipelines_in_core_waves_without_clashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(launcher.os, "sched_getaffinity", return_value={4, 5, 6}):
                rc = launcher.main(
                    [
                        "--pipeline",
                        "diagonal_4x4",
                        "--outputs-root",
                        str(outputs_root),
                        "--seeds",
                        "0",
                        "1",
                        "2",
                        "--cores",
                        "5",
                        "4",
                        "5",
                        "--max-parallel",
                        "2",
                        "--resume-policy",
                        "rerun_all",
                        "--shield-type",
                        "probabilistic",
                        "--shield-risk-threshold",
                        "0.25",
                        "--shield-theta",
                        "1e-8",
                        "--shield-max-vi-steps",
                        "250",
                        "--unsafe-cost-threshold",
                        "0.75",
                        "--success-rate",
                        "0.8",
                        "--dry-run",
                    ],
                )

            self.assertEqual(rc, 0)
            summary_path = outputs_root / "diagonal_4x4" / "multi_seed_logs" / "full_pipeline" / "summary.yaml"
            self.assertTrue(summary_path.exists())
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            jobs = summary["jobs"]
            self.assertEqual(summary["run_settings"]["core_pool"], [5, 4])
            self.assertEqual(summary["run_settings"]["shield_type"], "probabilistic")
            self.assertEqual(summary["run_settings"]["shield_risk_threshold"], 0.25)
            self.assertEqual(summary["run_settings"]["shield_theta"], 1e-8)
            self.assertEqual(summary["run_settings"]["shield_max_vi_steps"], 250)
            self.assertEqual(summary["run_settings"]["unsafe_cost_threshold"], 0.75)
            self.assertEqual(summary["run_settings"]["success_rate"], 0.8)
            self.assertEqual(len(jobs), 18)
            self.assertTrue(all(job["state"] == launcher.JOB_SUCCEEDED for job in jobs))
            self.assertEqual(
                [job["mode"] for job in jobs[:6]],
                [
                    "source",
                    "downstream_unconstrained",
                    "downstream_ewc",
                    "downstream_rashomon",
                    "downstream_safe_line_search",
                    "downstream_lagrangian",
                ],
            )
            self.assertEqual([job["core"] for job in jobs[:6]], [5, 5, 5, 5, 5, 5])
            self.assertEqual([job["core"] for job in jobs[6:12]], [4, 4, 4, 4, 4, 4])
            self.assertEqual([job["core"] for job in jobs[12:18]], [5, 5, 5, 5, 5, 5])
            self.assertEqual([job["scheduled_wave"] for job in jobs[:12]], [0] * 12)
            self.assertEqual([job["scheduled_wave"] for job in jobs[12:18]], [1] * 6)
            self.assertIn("--shield-type", jobs[0]["command"])
            self.assertIn("probabilistic", jobs[0]["command"])
            self.assertIn("--success-rate", jobs[0]["command"])
            self.assertIn("--success-rate", jobs[1]["command"])
            self.assertIn("0.8", jobs[1]["command"])
            self.assertNotIn("--shield-type", jobs[1]["command"])

            wave_to_seed_cores = {}
            for job in jobs:
                key = (job["scheduled_wave"], job["seed"])
                wave_to_seed_cores.setdefault(key, job["core"])
                self.assertEqual(wave_to_seed_cores[key], job["core"])
            for wave in {job["scheduled_wave"] for job in jobs}:
                cores_in_wave = {
                    core
                    for (scheduled_wave, _), core in wave_to_seed_cores.items()
                    if scheduled_wave == wave
                }
                seed_count = sum(1 for scheduled_wave, _ in wave_to_seed_cores if scheduled_wave == wave)
                self.assertEqual(len(cores_in_wave), seed_count)


if __name__ == "__main__":
    unittest.main()
