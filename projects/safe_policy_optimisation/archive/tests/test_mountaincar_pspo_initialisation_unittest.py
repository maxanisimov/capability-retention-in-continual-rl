"""Tests for sample-free MountainCar PSPO policy initialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from provably_safe_policy_optimisation.safe_init import (
    certify_with_verifier,
    refine_intervals_until_certified,
)
from torch import nn

from projects.safe_policy_optimisation.archive.stages.train_mountaincar_pspo_initialisation import (
    behavioural_clone_on_shield,
    build_parser,
    build_policy,
    critical_interval_tensors,
    run,
)


class IntervalRefinementTests(unittest.TestCase):
    def test_refines_complete_box_until_push_right_is_certified(self) -> None:
        model = nn.Sequential(nn.Linear(2, 3))
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].bias.copy_(torch.tensor([1.0, 0.0, -1.0]))
        x_l = torch.tensor([[-1.2, -0.07]])
        x_u = torch.tensor([[-1.0, 0.0]])
        safe_mask = torch.tensor([[False, False, True]])

        report = refine_intervals_until_certified(
            model,
            x_l,
            x_u,
            safe_mask,
            list(model.parameters()),
            lr=0.05,
            max_epochs=100,
            target_margin=0.1,
        )

        self.assertTrue(report.all_certified)
        self.assertEqual(report.certified_fraction, 1.0)
        self.assertGreaterEqual(report.final_ibp_margin, 0.1)
        self.assertGreater(report.epochs, 0)

    def test_failed_refinement_raises(self) -> None:
        model = nn.Sequential(nn.Linear(2, 3))
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].bias.copy_(torch.tensor([1.0, 0.0, -1.0]))
        with self.assertRaises(RuntimeError):
            refine_intervals_until_certified(
                model,
                torch.tensor([[-1.2, -0.07]]),
                torch.tensor([[-1.0, 0.0]]),
                torch.tensor([[False, False, True]]),
                list(model.parameters()),
                lr=0.05,
                max_epochs=0,
                target_margin=0.1,
            )


class MountainCarPSPOInitialisationStageTests(unittest.TestCase):
    def test_optional_bc_phase_clones_push_right_on_critical_samples(self) -> None:
        torch.manual_seed(1)
        model = build_policy(hidden_dim=8, n_hidden=1)
        report = behavioural_clone_on_shield(
            model,
            critical_min_position=-1.2,
            critical_max_position=-0.9,
            min_velocity=-0.07,
            max_velocity=0.0,
            n_samples=512,
            epochs=5,
            learning_rate=0.05,
            seed=1,
            device=torch.device("cpu"),
        )
        self.assertGreater(report["critical_samples"], 0)
        self.assertEqual(report["sampled_greedy_safe_rate"], 1.0)

    def test_interval_is_one_complete_box_not_samples(self) -> None:
        x_l, x_u, mask = critical_interval_tensors(
            critical_min_position=-1.2,
            critical_max_position=-1.0,
            min_velocity=-0.07,
            max_velocity=0.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(x_l.tolist(), [[-1.2000000476837158, -0.07000000029802322]])
        self.assertEqual(x_u.tolist(), [[-1.0, 0.0]])
        self.assertEqual(mask.int().tolist(), [[0, 0, 1]])

    def test_stage_saves_only_a_certified_compatible_base_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "--output-dir",
                    tmpdir,
                    "--run-id",
                    "smoke",
                    "--seed",
                    "0",
                    "--hidden-dim",
                    "8",
                    "--n-hidden",
                    "1",
                    "--learning-rate",
                    "0.05",
                    "--max-epochs",
                    "200",
                ]
            )
            summary = run(args)
            run_dir = Path(tmpdir) / "smoke"
            self.assertTrue((run_dir / "base_policy.pt").exists())
            self.assertTrue((run_dir / "critical_interval_dataset.pt").exists())
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue(summary["final_verification"]["all_certified"])

            payload = torch.load(
                run_dir / "base_policy.pt", map_location="cpu", weights_only=False
            )
            model = build_policy(hidden_dim=8, n_hidden=1)
            model.load_state_dict(payload["state_dict"])
            dataset = torch.load(
                run_dir / "critical_interval_dataset.pt",
                map_location="cpu",
                weights_only=False,
            )
            x_l, x_u, mask = dataset.tensors
            fraction, certified = certify_with_verifier(model, x_l, x_u, mask.bool())
            self.assertTrue(certified)
            self.assertEqual(fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
