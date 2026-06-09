"""Unit tests for the FrozenLake slippery shield safety adaptation-only launcher."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.frozenlake_slippery_shield_safety.cli import launch_adaptation_multi_seed as launcher


class FrozenLakeSafetyAdaptationLauncherTests(unittest.TestCase):
    def test_dry_run_schedules_runs_in_core_waves_without_clashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(launcher.os, "sched_getaffinity", return_value={4, 5, 6}):
                rc = launcher.main(
                    [
                        "--mode",
                        "downstream_rashomon",
                        "--pipeline",
                        "diagonal_4x4",
                        "--outputs-root",
                        str(outputs_root),
                        "--seeds",
                        "0",
                        "1",
                        "2",
                        "3",
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
            summary_path = (
                outputs_root
                / "diagonal_4x4"
                / "multi_seed_logs"
                / "adaptation_parallel"
                / "downstream_rashomon"
                / "summary.yaml"
            )
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            jobs = summary["jobs"]

            self.assertEqual(summary["run_settings"]["core_pool"], [5, 4])
            self.assertEqual([job["state"] for job in jobs], [launcher.JOB_SUCCEEDED] * 4)
            self.assertEqual([job["core"] for job in jobs], [5, 4, 5, 4])
            self.assertEqual([job["scheduled_wave"] for job in jobs], [0, 0, 1, 1])

    def test_rejects_max_parallel_larger_than_unique_core_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(launcher.os, "sched_getaffinity", return_value={0}):
                with self.assertRaisesRegex(ValueError, "exceeds selected unique core count"):
                    launcher.main(
                        [
                            "--mode",
                            "downstream_ewc",
                            "--outputs-root",
                            tmp_dir,
                            "--seeds",
                            "0",
                            "--cores",
                            "0",
                            "--max-parallel",
                            "2",
                            "--dry-run",
                        ],
                    )

    def test_all_completed_runs_are_skipped_and_still_write_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            run_dir = outputs_root / "diagonal_4x4" / "seed_0" / "downstream_unconstrained"
            run_dir.mkdir(parents=True)
            for artifact in launcher.MODE_TO_REQUIRED_ARTIFACTS["downstream_unconstrained"]:
                (run_dir / artifact).write_text("present", encoding="utf-8")

            with patch.object(launcher.os, "sched_getaffinity", return_value={0}):
                rc = launcher.main(
                    [
                        "--mode",
                        "downstream_unconstrained",
                        "--outputs-root",
                        str(outputs_root),
                        "--seeds",
                        "0",
                    ],
                )

            self.assertEqual(rc, 0)
            summary_path = (
                outputs_root
                / "diagonal_4x4"
                / "multi_seed_logs"
                / "adaptation_parallel"
                / "downstream_unconstrained"
                / "summary.yaml"
            )
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["jobs"][0]["state"], launcher.JOB_SKIPPED)
            self.assertIsNone(summary["jobs"][0]["core"])

    def test_dry_run_forwards_safe_line_search_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(launcher.os, "sched_getaffinity", return_value={0}):
                rc = launcher.main(
                    [
                        "--mode",
                        "downstream_safe_line_search",
                        "--outputs-root",
                        str(outputs_root),
                        "--seeds",
                        "0",
                        "--inverse-temp-start",
                        "3",
                        "--inverse-temp-max",
                        "7",
                        "--safe-line-search-max-backtracks",
                        "4",
                        "--safe-line-search-backtrack-coef",
                        "0.25",
                        "--success-rate",
                        "0.8",
                        "--dry-run",
                    ],
                )

            self.assertEqual(rc, 0)
            summary_path = (
                outputs_root
                / "diagonal_4x4"
                / "multi_seed_logs"
                / "adaptation_parallel"
                / "downstream_safe_line_search"
                / "summary.yaml"
            )
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            command = summary["jobs"][0]["command"]
            self.assertIn("--inverse-temp-start", command)
            self.assertIn("3", command)
            self.assertIn("--safe-line-search-max-backtracks", command)
            self.assertIn("4", command)
            self.assertIn("--success-rate", command)
            self.assertIn("0.8", command)
            self.assertEqual(summary["run_settings"]["success_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()
