"""Unit tests for the LavaCrossing shield-safety adaptation launcher."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.safety.lavacrossing.cli import launch_adaptation_multi_seed as launcher


class LavaCrossingAdaptationLauncherTests(unittest.TestCase):
    def test_dry_run_forwards_method_and_slip_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(launcher.os, "sched_getaffinity", return_value={0}):
                rc = launcher.main(
                    [
                        "--mode",
                        "downstream_safe_line_search",
                        "--pipeline",
                        "corridor_7x7_deterministic",
                        "--outputs-root",
                        str(outputs_root),
                        "--seeds",
                        "0",
                        "--inverse-temp-start",
                        "3",
                        "--safe-line-search-max-backtracks",
                        "4",
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
                / "adaptation_parallel"
                / "downstream_safe_line_search"
                / "summary.yaml"
            )
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            command = summary["jobs"][0]["command"]

            self.assertIn("--inverse-temp-start", command)
            self.assertIn("--safe-line-search-max-backtracks", command)
            self.assertIn("--slip-prob", command)
            self.assertIn("0.1", command)
            self.assertEqual(summary["run_settings"]["slip_prob"], 0.1)


if __name__ == "__main__":
    unittest.main()
