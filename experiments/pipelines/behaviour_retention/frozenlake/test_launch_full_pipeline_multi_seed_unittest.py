"""Unit tests for FrozenLake full-pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.behaviour_retention.frozenlake.core.orchestration import launch_full_pipeline_multi_seed as fp


class FrozenLakeFullPipelineGraphTests(unittest.TestCase):
    def test_source_success_unlocks_downstream_jobs(self) -> None:
        jobs = fp._create_job_graph([0])
        fp._refresh_ready_states(jobs)
        self.assertEqual(jobs["source:0"].state, fp.JOB_READY)
        for mode in fp.DOWNSTREAM_MODES:
            self.assertEqual(jobs[f"{mode}:0"].state, fp.JOB_PENDING)

        jobs["source:0"].state = fp.JOB_SUCCEEDED
        fp._refresh_ready_states(jobs)
        for mode in fp.DOWNSTREAM_MODES:
            self.assertEqual(jobs[f"{mode}:0"].state, fp.JOB_READY)

    def test_source_failure_blocks_downstream_jobs(self) -> None:
        jobs = fp._create_job_graph([0])
        fp._refresh_ready_states(jobs)
        jobs["source:0"].state = fp.JOB_FAILED
        fp._refresh_ready_states(jobs)
        for mode in fp.DOWNSTREAM_MODES:
            self.assertEqual(jobs[f"{mode}:0"].state, fp.JOB_BLOCKED)


class FrozenLakeFullPipelineDryRunTests(unittest.TestCase):
    def test_dry_run_writes_summary_for_all_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(fp.os, "sched_getaffinity", return_value={0, 1}):
                rc = fp.main(
                    [
                        "--pipeline",
                        "dryrun_unit_case",
                        "--outputs-root",
                        str(outputs_root),
                        "--seeds",
                        "0",
                        "1",
                        "--cores",
                        "0",
                        "1",
                        "--dry-run",
                        "--no-aggregate-metrics",
                    ],
                )

            self.assertEqual(rc, 0)
            summary_path = outputs_root / "dryrun_unit_case" / "multi_seed_logs" / "full_pipeline" / "summary.yaml"
            self.assertTrue(summary_path.exists())
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
            jobs = summary.get("jobs", [])
            self.assertEqual(len(jobs), 8)
            self.assertTrue(all(job["state"] == fp.JOB_SUCCEEDED for job in jobs))


if __name__ == "__main__":
    unittest.main()
