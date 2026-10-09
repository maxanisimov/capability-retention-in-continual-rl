"""Combined variant retains reference settings and initial policies."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pspo_verify_first_segment as launcher

from projects.safe_policy_optimisation.stages import train_pspo


class VerifyFirstSegmentLauncherTests(unittest.TestCase):
    def test_all_60_jobs_preserve_reference_protocol(self):
        manifest = launcher.build_manifest(
            Path("/tmp/verify-first-segment-test"), list(range(8, 68))
        )
        self.assertEqual(len(manifest["jobs"]), 60)
        for job in manifest["jobs"]:
            args = train_pspo.parse_args(job["command"][5:])
            source = launcher.read_json(Path(job["source_config"]))
            self.assertTrue(args.verify_first)
            self.assertEqual(args.safe_region_shape, "segment")
            self.assertEqual(str(args.base_policy_path), source["base_policy_path"])
            self.assertEqual(args.seed, job["seed"])
            self.assertEqual(args.total_timesteps, source["total_timesteps"])
            self.assertEqual(args.eval_episodes, 100)
            for key, value in source["training_hyperparameters"].items():
                self.assertEqual(getattr(args, key), value)
            for key in ["segment_tolerance", "segment_splits", "segment_max_splits"]:
                self.assertEqual(getattr(args, key), source["adaptive"][key])

    def test_smoke_has_six_tiny_jobs(self):
        manifest = launcher.build_manifest(
            Path("/tmp/verify-first-segment-smoke-test"), list(range(8, 14)), smoke=True
        )
        self.assertEqual(len(manifest["jobs"]), 6)
        self.assertTrue(all(job["nominal_timesteps"] == 8 for job in manifest["jobs"]))

    def test_rejects_shared_cpus(self):
        with self.assertRaises(ValueError):
            launcher.build_manifest(
                Path("/tmp/verify-first-segment-invalid-test"), [8] * 60
            )


if __name__ == "__main__":
    unittest.main()
