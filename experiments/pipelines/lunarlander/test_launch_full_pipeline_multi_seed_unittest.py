"""Unit tests for full-pipeline multi-seed orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.lunarlander.core.orchestration import launch_full_pipeline_multi_seed as fp


class _FakeSuccessProcess:
    _next_pid = 40_000

    def __init__(self) -> None:
        self.pid = _FakeSuccessProcess._next_pid
        _FakeSuccessProcess._next_pid += 1

    def poll(self) -> int:
        return 0


class LaunchFullPipelineGraphTests(unittest.TestCase):
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

    def test_continue_failure_policy_does_not_block_other_seed(self) -> None:
        jobs = fp._create_job_graph([0, 1])
        fp._refresh_ready_states(jobs)

        jobs["source:0"].state = fp.JOB_FAILED
        stop_launching = fp._apply_failure_policy(
            jobs,
            failed_job=jobs["source:0"],
            failure_policy="continue",
        )
        fp._refresh_ready_states(jobs)

        self.assertFalse(stop_launching)
        self.assertEqual(jobs["source:1"].state, fp.JOB_READY)
        self.assertEqual(jobs["downstream_unconstrained:1"].state, fp.JOB_PENDING)
        for mode in fp.DOWNSTREAM_MODES:
            self.assertEqual(jobs[f"{mode}:0"].state, fp.JOB_BLOCKED)

    def test_balanced_dispatch_can_overlap_source_and_downstream(self) -> None:
        source_job = fp.Job(seed=1, mode="source", state=fp.JOB_READY)
        downstream_job = fp.Job(seed=0, mode="downstream_unconstrained", state=fp.JOB_READY)

        selected = fp._select_jobs_to_launch(
            ready_sources=[source_job],
            ready_downstream=[downstream_job],
            free_slots=2,
            dispatch_policy="balanced",
            total_cores=2,
            active_source_count=1,
            sources_remaining=True,
        )

        self.assertEqual({job.mode for job in selected}, {"source", "downstream_unconstrained"})


class LaunchFullPipelineResumeTests(unittest.TestCase):
    def test_completion_detection_and_skip_completed_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            task_setting = "resume_case"

            source_dir = fp._job_output_dir(outputs_root, task_setting, 0, "source")
            source_dir.mkdir(parents=True, exist_ok=True)
            for name in fp.MODE_TO_REQUIRED_ARTIFACTS["source"]:
                (source_dir / name).write_text("ok", encoding="utf-8")
            self.assertTrue(fp._is_job_complete(outputs_root, task_setting, 0, "source"))

            ewc_dir = fp._job_output_dir(outputs_root, task_setting, 0, "downstream_ewc")
            ewc_dir.mkdir(parents=True, exist_ok=True)
            for name in fp.MODE_TO_REQUIRED_ARTIFACTS["downstream_ewc"][:-1]:
                (ewc_dir / name).write_text("ok", encoding="utf-8")
            self.assertFalse(fp._is_job_complete(outputs_root, task_setting, 0, "downstream_ewc"))
            (ewc_dir / "ewc_state.pt").write_text("ok", encoding="utf-8")
            self.assertTrue(fp._is_job_complete(outputs_root, task_setting, 0, "downstream_ewc"))

            unconstrained_dir = fp._job_output_dir(
                outputs_root,
                task_setting,
                0,
                "downstream_unconstrained",
            )
            unconstrained_dir.mkdir(parents=True, exist_ok=True)
            for name in fp.MODE_TO_REQUIRED_ARTIFACTS["downstream_unconstrained"]:
                (unconstrained_dir / name).write_text("ok", encoding="utf-8")

            jobs = fp._create_job_graph([0])
            args = argparse.Namespace(
                resume_policy="skip_completed",
                outputs_root=outputs_root,
                task_setting=task_setting,
            )
            fp._apply_resume_policy(jobs, args)
            fp._refresh_ready_states(jobs)

            self.assertEqual(jobs["source:0"].state, fp.JOB_SKIPPED)
            self.assertEqual(jobs["downstream_unconstrained:0"].state, fp.JOB_SKIPPED)
            self.assertEqual(jobs["downstream_ewc:0"].state, fp.JOB_SKIPPED)
            self.assertEqual(jobs["downstream_rashomon:0"].state, fp.JOB_READY)


class LaunchFullPipelineIntegrationTests(unittest.TestCase):
    def test_main_dry_run_two_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(fp.os, "sched_getaffinity", return_value={0, 1}):
                with patch.object(fp.subprocess, "run") as aggregate_run:
                    rc = fp.main(
                        [
                            "--task-setting",
                            "dryrun_case",
                            "--outputs-root",
                            str(outputs_root),
                            "--seeds",
                            "0",
                            "1",
                            "--cores",
                            "0",
                            "1",
                            "--poll-seconds",
                            "0",
                            "--dry-run",
                        ],
                    )
                    aggregate_run.assert_not_called()

            self.assertEqual(rc, 0)
            summary_path = outputs_root / "dryrun_case" / "multi_seed_logs" / "full_pipeline" / "summary.yaml"
            self.assertTrue(summary_path.exists())
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
            jobs = summary.get("jobs", [])
            self.assertEqual(len(jobs), 8)
            self.assertTrue(all(j["state"] == fp.JOB_SUCCEEDED for j in jobs))

    def test_main_smoke_non_dry_with_mocked_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            with patch.object(fp.os, "sched_getaffinity", return_value={0}):
                with patch.object(
                    fp,
                    "_spawn_subprocess",
                    side_effect=lambda *_, **__: _FakeSuccessProcess(),
                ):
                    with patch.object(
                        fp.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess(args=[], returncode=0),
                    ) as aggregate_run:
                        rc = fp.main(
                            [
                                "--task-setting",
                                "smoke_case",
                                "--outputs-root",
                                str(outputs_root),
                                "--seeds",
                                "0",
                                "--cores",
                                "0",
                                "--poll-seconds",
                                "0",
                            ],
                        )
                        aggregate_run.assert_called_once()
                        cmd = list(aggregate_run.call_args.args[0])
                        self.assertTrue(any("aggregate_layout_metrics.py" in token for token in cmd))
                        self.assertIn("smoke_case", cmd)

            self.assertEqual(rc, 0)
            log_root = outputs_root / "smoke_case" / "multi_seed_logs" / "full_pipeline"
            for mode in fp.ALL_MODES:
                self.assertTrue((log_root / mode / "seed_0.log").exists())
            summary_path = log_root / "summary.yaml"
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
