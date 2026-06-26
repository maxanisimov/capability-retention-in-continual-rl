"""Unit tests for the LunarLander multi-seed launcher."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from projects.safe_crl.pipelines._shared import multi_seed_launcher as shared_launcher
from projects.safe_crl.pipelines.trajectory_retention.lunarlander.cli import launch_multi_seed as launcher


class LunarLanderLaunchMultiSeedTests(unittest.TestCase):
    def test_max_parallel_truncates_core_pool(self) -> None:
        core_pool = launcher._apply_max_parallel([5, 4, 6], 2)
        self.assertEqual(core_pool, [5, 4])

    def test_max_parallel_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            launcher._apply_max_parallel([5, 4], 0)
        with self.assertRaises(ValueError):
            launcher._apply_max_parallel([5, 4], 5)

    def test_build_cmd_forwards_methods_and_run_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = launcher._parse_args(
                [
                    "--pipeline", "deterministic_test",
                    "--methods", "unconstrained", "rashomon",
                    "--seeds", "0",
                    "--outputs-root", tmp_dir,
                ],
            )
            cmd = launcher.build_cmd(args, 0)

            self.assertIn("run_seed_pipeline.py", cmd[1])
            self.assertEqual(cmd[cmd.index("--methods") + 1 : cmd.index("--methods") + 3], ["unconstrained", "rashomon"])
            self.assertEqual(cmd[cmd.index("--seed") + 1], "0")
            self.assertIn("--rl", cmd)
            self.assertEqual(cmd[cmd.index("--rl") + 1], "ppo")

    def test_dry_run_prints_one_command_per_seed_without_scheduling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(shared_launcher.os, "sched_getaffinity", return_value={0, 1, 2}):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    rc = launcher.main(
                        [
                            "--pipeline", "deterministic_test",
                            "--methods", "unconstrained",
                            "--seeds", "0", "1", "2",
                            "--cores", "0", "1",
                            "--outputs-root", str(outputs_root),
                            "--dry-run",
                        ],
                    )
            self.assertEqual(rc, 0)
            output = stdout.getvalue()
            for seed in (0, 1, 2):
                self.assertIn(f"[dry-run] seed={seed}", output)
            self.assertIn("run_seed_pipeline.py", output)


if __name__ == "__main__":
    unittest.main()
