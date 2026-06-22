"""Unit tests for the FrozenLake safety full-pipeline launcher."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.safety_retention.frozenlake.cli import launch_full_pipeline_multi_seed as launcher


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
                        "--dry-run",
                    ],
                )

            self.assertEqual(rc, 0)
            summary_path = outputs_root / "diagonal_4x4" / "multi_seed_logs" / "full_pipeline" / "summary.yaml"
            self.assertTrue(summary_path.exists())
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            jobs = summary["jobs"]
            self.assertEqual(summary["run_settings"]["core_pool"], [5, 4])
            self.assertEqual(len(jobs), 12)
            self.assertTrue(all(job["state"] == launcher.JOB_SUCCEEDED for job in jobs))
            self.assertEqual(
                [job["mode"] for job in jobs[:4]],
                ["source", "downstream_unconstrained", "downstream_ewc", "downstream_rashomon"],
            )
            self.assertEqual([job["core"] for job in jobs[:4]], [5, 5, 5, 5])
            self.assertEqual([job["core"] for job in jobs[4:8]], [4, 4, 4, 4])
            self.assertEqual([job["core"] for job in jobs[8:12]], [5, 5, 5, 5])
            self.assertEqual([job["scheduled_wave"] for job in jobs[:8]], [0] * 8)
            self.assertEqual([job["scheduled_wave"] for job in jobs[8:12]], [1] * 4)

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
