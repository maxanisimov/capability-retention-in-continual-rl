"""Tests for the continuous-state PSPO MountainCar stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from continuous_state_shields import MountainCarIntervalBoxShield, MountainCarShield
from torch.utils.data import TensorDataset

from projects.safe_policy_optimisation.stages.train_pspo_continuous import (
    load_interval_certificate,
    parse_args,
    run,
    validate_mountaincar_certificate,
)


def _write_safe_inputs(directory: Path) -> tuple[Path, Path]:
    base_path = directory / "base_policy.pt"
    certificate_path = directory / "critical_interval_dataset.pt"
    torch.save(
        {
            "architecture": {
                "input_dim": 2,
                "n_actions": 3,
                "hidden_dim": 8,
                "n_hidden": 0,
                "activation": "Tanh",
                "state_representation": "continuous_features",
            },
            "state_dict": {
                "0.weight": torch.zeros((3, 2)),
                "0.bias": torch.tensor([0.0, 0.0, 5.0]),
            },
        },
        base_path,
    )
    torch.save(
        TensorDataset(
            torch.tensor([[-1.2, -0.07]]),
            torch.tensor([[-1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
        ),
        certificate_path,
    )
    return base_path, certificate_path


class ContinuousPSPOStageTests(unittest.TestCase):
    def test_defaults_to_region_first_mountaincar_and_shaped_reward(self) -> None:
        args = parse_args(["--base-policy-path", "/tmp/base_policy.pt"])
        self.assertEqual(args.env_id, "MountainCar-v0")
        self.assertFalse(args.verify_first)
        self.assertTrue(args.mountaincar_shaped_reward)
        self.assertEqual(args.safe_region_shape, "orthotope")

    def test_rejects_zonotope_input_interval_certification(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--base-policy-path",
                    "/tmp/base_policy.pt",
                    "--safe-region-shape",
                    "zonotope",
                    "--directional",
                    "false",
                ]
            )

    def test_accepts_segment_safe_region(self) -> None:
        args = parse_args(
            [
                "--base-policy-path",
                "/tmp/base_policy.pt",
                "--safe-region-shape",
                "segment",
            ]
        )
        self.assertEqual(args.safe_region_shape, "segment")
        self.assertEqual(args.segment_tolerance, 1e-3)
        self.assertEqual(args.segment_splits, 4)
        self.assertEqual(args.segment_max_splits, 8)

    def test_rejects_segment_combinations_the_core_cannot_honour(self) -> None:
        """Fail at the CLI rather than with a traceback mid-training."""
        for label, extra in (
            ("freq once", ["--freq", "once", "--verify-first", "false",
                           "--directional", "false"]),
            ("region-refresh fixed", ["--region-refresh", "fixed",
                                      "--directional", "false"]),
            ("region-mode union", ["--region-mode", "union"]),
            # The segment engine calls bound_forward_pass directly, so a
            # non-IBP growth method would be silently ignored.
            ("growth-method CROWN", ["--growth-method", "CROWN"]),
        ):
            with self.subTest(label), self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--base-policy-path",
                        "/tmp/base_policy.pt",
                        "--safe-region-shape",
                        "segment",
                        *extra,
                    ]
                )

    def test_certificate_must_match_runtime_shield(self) -> None:
        dataset = TensorDataset(
            torch.tensor([[-1.19, -0.07]]),
            torch.tensor([[-1.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
        )
        with self.assertRaisesRegex(ValueError, "lower bound"):
            validate_mountaincar_certificate(dataset, MountainCarShield())

    def test_box_certificate_must_match_boxed_runtime_shield(self) -> None:
        lows = np.asarray([[-1.2, -0.07], [-0.8, -0.02]], dtype=np.float32)
        highs = np.asarray([[-1.0, 0.0], [-0.6, 0.02]], dtype=np.float32)
        masks = np.asarray([[0, 0, 1], [0, 1, 1]], dtype=bool)
        shield = MountainCarIntervalBoxShield(
            box_lows=lows, box_highs=highs, safe_masks=masks
        )
        dataset = TensorDataset(
            torch.from_numpy(lows),
            torch.from_numpy(highs),
            torch.from_numpy(masks.astype(np.float32)),
        )
        validate_mountaincar_certificate(dataset, shield)
        bad = TensorDataset(
            torch.from_numpy(lows),
            torch.from_numpy(highs),
            torch.ones((2, 3)),
        )
        with self.assertRaisesRegex(ValueError, "action masks"):
            validate_mountaincar_certificate(bad, shield)

    def test_one_update_runs_pass_final_certificate_in_both_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path, certificate_path = _write_safe_inputs(root)
            loaded = load_interval_certificate(certificate_path, "MountainCar-v0")
            self.assertEqual(len(loaded), 1)
            for verify_first in (False, True):
                run_id = "verify_first" if verify_first else "region_first"
                args = parse_args(
                    [
                        "--base-policy-path",
                        str(base_path),
                        "--certificate-dataset",
                        str(certificate_path),
                        "--verify-first",
                        str(verify_first).lower(),
                        "--total-timesteps",
                        "8",
                        "--curve-eval-freq",
                        "0",
                        "--eval-episodes",
                        "1",
                        "--n-steps",
                        "8",
                        "--batch-size",
                        "8",
                        "--n-epochs",
                        "1",
                        "--n-iters",
                        "1",
                        "--rashomon-checkpoint",
                        "1",
                        "--output-dir",
                        str(root),
                        "--run-id",
                        run_id,
                    ]
                )
                summary = run(args)
                self.assertTrue(summary["final_interval_certified"])
                self.assertEqual(summary["final_interval_certified_fraction"], 1.0)
                self.assertTrue((root / run_id / "model.zip").exists())

    def test_one_update_runs_with_a_segment_safe_region(self) -> None:
        """The segment path certifies end to end against an interval certificate."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path, certificate_path = _write_safe_inputs(root)
            args = parse_args(
                [
                    "--base-policy-path",
                    str(base_path),
                    "--certificate-dataset",
                    str(certificate_path),
                    "--safe-region-shape",
                    "segment",
                    "--total-timesteps",
                    "8",
                    "--curve-eval-freq",
                    "0",
                    "--eval-episodes",
                    "1",
                    "--n-steps",
                    "8",
                    "--batch-size",
                    "8",
                    "--n-epochs",
                    "1",
                    "--n-iters",
                    "1",
                    "--rashomon-checkpoint",
                    "1",
                    "--output-dir",
                    str(root),
                    "--run-id",
                    "segment",
                ]
            )
            summary = run(args)
            self.assertTrue(summary["final_interval_certified"])
            diagnostics = summary["adaptive_diagnostics"]
            self.assertEqual(diagnostics["safe_region_shape"], "segment")
            self.assertEqual(diagnostics["segment_tolerance"], 1e-3)
            self.assertEqual(diagnostics["segment_splits"], 4)
            self.assertEqual(diagnostics["segment_max_splits"], 8)
            self.assertTrue((root / "segment" / "model.zip").exists())


if __name__ == "__main__":
    unittest.main()
