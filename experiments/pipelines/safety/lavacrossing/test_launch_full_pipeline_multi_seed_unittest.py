"""Unit tests for the LavaCrossing shield-safety full-pipeline launcher."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.safety.lavacrossing.cli import launch_full_pipeline_multi_seed as launcher


class LavaCrossingFullLauncherTests(unittest.TestCase):
    def test_dry_run_schedules_all_modes_and_forwards_slip_prob(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(launcher.os, "sched_getaffinity", return_value={4, 5, 6}):
                rc = launcher.main(
                    [
                        "--pipeline",
                        "corridor_7x7_deterministic",
                        "--outputs-root",
                        str(outputs_root),
                        "--seeds",
                        "0",
                        "1",
                        "--cores",
                        "5",
                        "4",
                        "--max-parallel",
                        "2",
                        "--resume-policy",
                        "rerun_all",
                        "--shield-type",
                        "probabilistic",
                        "--shield-risk-threshold",
                        "0.25",
                        "--slip-prob",
                        "0.1",
                        "--dry-run",
                    ],
                )

            self.assertEqual(rc, 0)
            summary_path = (
                outputs_root
                / "corridor_7x7_deterministic"
                / "multi_seed_logs"
                / "full_pipeline"
                / "summary.yaml"
            )
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            jobs = summary["jobs"]

            self.assertEqual(summary["run_settings"]["core_pool"], [5, 4])
            self.assertEqual(summary["run_settings"]["slip_prob"], 0.1)
            self.assertEqual(len(jobs), 12)
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
            self.assertTrue(all(job["state"] == launcher.JOB_SUCCEEDED for job in jobs))
            self.assertIn("--slip-prob", jobs[0]["command"])
            self.assertIn("--slip-prob", jobs[1]["command"])
            self.assertIn("--shield-type", jobs[0]["command"])
            self.assertNotIn("--shield-type", jobs[1]["command"])


if __name__ == "__main__":
    unittest.main()
