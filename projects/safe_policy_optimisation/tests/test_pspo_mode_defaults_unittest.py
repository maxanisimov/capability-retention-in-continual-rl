"""Check that geometry and enforcement choices reach the training workers."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from projects.safe_policy_optimisation.scripts.experiments import (
    launch_pspo_multi_env as masa,
    run_frozenlake_pspo_shaping as frozenlake,
    run_frozenlake_shaping_sweep as frozenlake_sweep,
    run_inference_time_comparison as inference,
    run_pspo_four_ablations as ablations,
    run_temperature_sweep as temperature,
    run_timed_main_comparison as timing,
)
from projects.safe_policy_optimisation.stages import train_pspo


class PSPOModeTests(unittest.TestCase):
    def test_evaluators_select_matching_saved_modes(self):
        for shape in ("segment", "orthotope"):
            for verify in (False, True):
                manifest = inference.build_manifest(
                    Path("/tmp/inference-mode-test"), [0], ["media_streaming"],
                    10, 0, pspo_variant=shape, verify_first=verify,
                )
                job = next(job for job in manifest["jobs"] if job["method"] == "pspo")
                adaptive = job["config"]["adaptive"]
                self.assertEqual(adaptive.get("safe_region_shape", "orthotope"), shape)
                self.assertEqual(adaptive.get("verify_first", False), verify)
        self.assertIn("pspo_ls", temperature.DEFAULT_VARIANT_KEYS)
        self.assertNotIn("pspo", temperature.DEFAULT_VARIANT_KEYS)
        self.assertIn("pspo", temperature.VARIANT_BY_KEY)
        self.assertIn("pspo_ls_verify_first", temperature.VARIANT_BY_KEY)

    def test_masa_default_and_explicit_modes_reach_the_shell_worker(self):
        self.assertEqual(masa.build_parser().parse_args([]).safe_region_shape, "segment")
        for shape in ("segment", "orthotope"):
            for verify in (False, True):
                with self.subTest(shape=shape, verify=verify):
                    args = masa.build_parser().parse_args(
                        ["--safe-region-shape", shape, "--verify-first", str(verify).lower()]
                    )
                    env = masa.build_launch_environment(
                        environment="media_streaming", seeds=[0], cpu_ids=[0],
                        architecture="two_hidden", run_name="mode-test", n_iters=1,
                        dry_run=True, safe_region_shape=args.safe_region_shape,
                        verify_first=args.verify_first == "true",
                    )
                    self.assertEqual(env["SAFE_REGION_SHAPE"], shape)
                    self.assertEqual(env["VERIFY_FIRST"], str(verify).lower())
        self.assertEqual(train_pspo.build_parser().parse_args(
            ["--base-policy-path", "unused", "--shield-path", "unused"]
        ).safe_region_shape, "segment")

    def test_timing_overrides_historical_geometry_with_the_requested_mode(self):
        for shape, verify in (("segment", False), ("orthotope", True)):
            manifest = timing.build_manifest(
                Path("/tmp/pspo-mode-timing-test"), [0], ["media_streaming"], False,
                safe_region_shape=shape, verify_first=verify,
            )
            for job in manifest["jobs"]:
                if job["kind"] != "training" or not job["method"].startswith("pspo"):
                    continue
                values = timing.training_values(job, Path("/tmp/mode-test/seed0"), 0, False)
                args = train_pspo.parse_args(timing.cli_arguments(train_pspo.build_parser(), values))
                self.assertEqual(args.safe_region_shape, shape)
                self.assertEqual(args.verify_first, verify)

    def test_frozenlake_sweep_forwards_each_mode_to_its_workers(self):
        for shape in ("segment", "orthotope"):
            for verify in (False, True):
                with self.subTest(shape=shape, verify=verify), tempfile.TemporaryDirectory() as tmp:
                    settings = argparse.Namespace(
                        output_dir=Path(tmp) / "new", size=16, total_timesteps=16,
                        eval_episodes=2, screen_session="mode-test", smoke=True,
                        methods=["pspo"], safe_region_shape=shape, verify_first=verify,
                    )
                    manifest = frozenlake_sweep.build_manifest(settings, [0])
                    command = manifest["jobs"][0]["command"]
                    args = frozenlake.build_parser().parse_args(command[command.index("-u") + 2:])
                    self.assertEqual(args.safe_region_shape, shape)
                    self.assertEqual(args.verify_first, verify)
                    self.assertEqual(manifest["safe_region_shape"], shape)
                    self.assertEqual(manifest["verify_first"], verify)

    def test_segment_ablation_uses_its_own_initialiser(self):
        job = ablations.variant_environment(
            "no_margin", "media_streaming", 0, cpu=0, architecture="two_hidden",
            source={"base_policy_path": "/tmp/control.pt"},
            output_root=Path("/tmp/ablation-mode-test"), dry_run=True,
            total_timesteps_override=None,
        )
        self.assertEqual(job.env["SAFE_REGION_SHAPE"], "segment")
        self.assertEqual(job.env["BC_MARGIN_LOSS_WEIGHT"], "0.0")
        self.assertNotEqual(job.env.get("BASE_POLICY_PATH"), "/tmp/control.pt")
        with self.assertRaisesRegex(ValueError, "orthotope ablation"):
            ablations.variant_environment(
                "fixed_lid", "media_streaming", 0, cpu=0, architecture="two_hidden",
                source=None, output_root=Path("/tmp/ablation-mode-test"),
                dry_run=True, total_timesteps_override=None,
            )

    def test_all_frozenlake_modes_produce_matching_safe_checkpoints(self):
        for shape in ("segment", "orthotope"):
            for verify in (False, True):
                with self.subTest(shape=shape, verify=verify), tempfile.TemporaryDirectory() as tmp:
                    args = frozenlake.build_parser().parse_args([
                        "--output-dir", tmp, "--run-id", "smoke", "--size", "16",
                        "--total-timesteps", "16", "--n-steps", "8", "--batch-size", "8",
                        "--n-epochs", "1", "--max-episode-steps", "8",
                        "--eval-episodes", "2", "--curve-eval-freq", "0",
                        "--curve-eval-episodes", "2", "--lid-iters", "1",
                        "--lid-checkpoint", "1", "--lid-batch-size", "64", "--verbose", "0",
                        "--safe-region-shape", shape, "--verify-first", str(verify).lower(),
                    ])
                    with contextlib.redirect_stdout(io.StringIO()):
                        summary = frozenlake.run(args)
                    config = json.loads((Path(tmp) / "smoke/config.json").read_text())
                    self.assertEqual(config["safety_enforcement"]["safe_region_shape"], shape)
                    self.assertEqual(config["safety_enforcement"]["verify_first"], verify)
                    self.assertEqual(summary["adaptive_diagnostics"]["safe_region_shape"], shape)
                    self.assertEqual(summary["adaptive_diagnostics"]["verify_first"], verify)
                    self.assertEqual(summary["final_exact_all_state_alignment"], 1.0)


if __name__ == "__main__":
    unittest.main()
