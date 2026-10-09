"""Segment geometry, artifact provenance and certified raw evaluation tests."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from projects.safe_policy_optimisation.scripts import (
    run_frozenlake_segment_shaping as runner,
)


class FrozenLakeSegmentShapingTests(unittest.TestCase):
    def test_manifest_launches_ten_segment_workers_and_snapshots_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            args = type(
                "Args",
                (),
                {
                    "output_dir": Path(temp) / "sweep",
                    "size": 16,
                    "total_timesteps": 204800,
                    "eval_episodes": 100,
                    "screen_session": "test",
                    "smoke": False,
                    "methods": ("pspo",),
                },
            )()
            manifest = runner.prepare(args, list(range(10)))
            self.assertEqual(manifest["seeds"], list(range(10)))
            self.assertEqual(manifest["safe_region_shape"], "segment")
            self.assertFalse(manifest["verify_first"])
            for job in manifest["jobs"]:
                self.assertEqual(
                    job["command"][5], str(Path(runner.__file__).resolve())
                )
                self.assertIn("--worker", job["command"])
            relative = str(Path(runner.__file__).resolve().relative_to(runner.REPO))
            self.assertIn(relative, manifest["source_sha256"])
            with zipfile.ZipFile(args.output_dir / "source_snapshot.zip") as snapshot:
                self.assertIn(relative, snapshot.namelist())

    def test_worker_restores_adapters_on_failure(self):
        model, write, sources = (
            runner.pspo.AdaptiveSafePPOV2,
            runner.pspo.write_json,
            runner.pspo.source_paths,
        )
        args = runner.worker_parser().parse_args(["--output-dir", "/tmp"])
        with patch.object(runner.pspo, "run", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                runner.run_worker(args)
        self.assertIs(runner.pspo.AdaptiveSafePPOV2, model)
        self.assertIs(runner.pspo.write_json, write)
        self.assertIs(runner.pspo.source_paths, sources)

    def test_verify_first_manifest_forwards_flag_to_every_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            args = type(
                "Args",
                (),
                {
                    "output_dir": Path(temp) / "sweep",
                    "size": 32,
                    "total_timesteps": 204800,
                    "eval_episodes": 100,
                    "screen_session": "test",
                    "smoke": False,
                    "methods": ("pspo",),
                    "verify_first": True,
                },
            )()
            manifest = runner.prepare(args, list(range(10)))
            self.assertEqual(manifest["pspo_variant"], "line_segment_verify_first")
            self.assertTrue(manifest["verify_first"])
            self.assertEqual(len(manifest["jobs"]), 10)
            for job in manifest["jobs"]:
                self.assertIn("--verify-first", job["command"])

    def test_verify_first_smokes_accept_safe_proposals_without_computing_lid(self):
        for size in (16, 32):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temp:
                args = runner.worker_parser().parse_args(
                    [
                        "--worker",
                        "--verify-first",
                        "--size",
                        str(size),
                        "--output-dir",
                        temp,
                        "--run-id",
                        "smoke",
                        "--total-timesteps",
                        "64",
                        "--n-steps",
                        "32",
                        "--batch-size",
                        "16",
                        "--n-epochs",
                        "1",
                        "--max-episode-steps",
                        "16",
                        "--eval-episodes",
                        "2",
                        "--curve-eval-episodes",
                        "2",
                        "--curve-eval-freq",
                        "32",
                        "--lid-iters",
                        "1",
                        "--lid-checkpoint",
                        "1",
                        "--lid-batch-size",
                        "64",
                        "--verbose",
                        "0",
                    ]
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = runner.run_worker(args)
                root = Path(temp) / "smoke"
                config = json.loads((root / "config.json").read_text())
                self.assertEqual(config["pspo_variant"], "line_segment_verify_first")
                self.assertEqual(
                    config["safety_enforcement"]["implementation"], "AdaptiveSafePPO"
                )
                self.assertTrue(config["safety_enforcement"]["verify_first"])
                self.assertEqual(
                    config["safety_enforcement"]["safe_region_shape"], "segment"
                )
                diag = summary["adaptive_diagnostics"]
                self.assertEqual(diag["safe_region_shape"], "segment")
                self.assertEqual(diag["verifications_run"], 1)
                self.assertEqual(diag["accepted_without_rashomon"], 1)
                self.assertEqual(diag["rashomon_computations"], 0)
                self.assertEqual(diag["final_flushes"], 1)
                self.assertEqual(summary["final_exact_all_state_alignment"], 1)
                self.assertFalse(summary["metrics"]["reward_shaping_enabled"])
                self.assertEqual(
                    summary["metrics"]["evaluation_policy"], "greedy_unshielded"
                )
                init = config["initialization"]
                self.assertEqual(
                    init["actor_sha256_before_shaping"],
                    init["actor_sha256_after_shaping"],
                )
                for key in (
                    "reward_information_used",
                    "goal_information_used",
                    "witness_policy_used",
                ):
                    self.assertFalse(init["initial_actor"][key])

    def test_smoke_uses_segment_and_evaluates_certified_policy_without_shaping(self):
        with tempfile.TemporaryDirectory() as temp:
            args = runner.worker_parser().parse_args(
                [
                    "--worker",
                    "--output-dir",
                    temp,
                    "--run-id",
                    "smoke",
                    "--size",
                    "16",
                    "--total-timesteps",
                    "64",
                    "--n-steps",
                    "32",
                    "--batch-size",
                    "16",
                    "--n-epochs",
                    "1",
                    "--max-episode-steps",
                    "16",
                    "--eval-episodes",
                    "2",
                    "--curve-eval-episodes",
                    "2",
                    "--curve-eval-freq",
                    "32",
                    "--lid-iters",
                    "1",
                    "--lid-checkpoint",
                    "1",
                    "--lid-batch-size",
                    "64",
                    "--verbose",
                    "0",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                summary = runner.run_worker(args)
            root = Path(temp) / "smoke"
            config = json.loads((root / "config.json").read_text())
            self.assertEqual(config["pspo_variant"], "line_segment")
            self.assertEqual(
                config["safety_enforcement"]["safe_region_shape"], "segment"
            )
            self.assertFalse(config["safety_enforcement"]["verify_first"])
            diagnostics = summary["adaptive_diagnostics"]
            self.assertEqual(diagnostics["safe_region_shape"], "segment")
            self.assertFalse(diagnostics["audit_candidates_exactly"])
            self.assertEqual(summary["final_timesteps"], 64)
            self.assertEqual(summary["final_exact_all_state_alignment"], 1)
            self.assertEqual(diagnostics["final_flushes"], 1)
            self.assertFalse(summary["metrics"]["reward_shaping_enabled"])
            self.assertEqual(
                summary["metrics"]["evaluation_policy"], "greedy_unshielded"
            )
            init = config["initialization"]
            self.assertEqual(
                init["actor_sha256_before_shaping"], init["actor_sha256_after_shaping"]
            )
            for key in (
                "reward_information_used",
                "goal_information_used",
                "witness_policy_used",
            ):
                self.assertFalse(init["initial_actor"][key])
            with (root / "episodes.csv").open() as handle:
                episodes = list(csv.DictReader(handle))
            self.assertEqual(len(episodes), 2)
            for row in episodes:
                self.assertAlmostEqual(
                    float(row["total_reward"]),
                    float(row["success"] == "True") - 0.001 * int(row["length"]),
                )
                self.assertEqual(row["safe_trajectory"], "True")
            with zipfile.ZipFile(root / "source_snapshot.zip") as snapshot:
                self.assertIn(
                    str(Path(runner.__file__).resolve().relative_to(runner.REPO)),
                    snapshot.namelist(),
                )


if __name__ == "__main__":
    unittest.main()
