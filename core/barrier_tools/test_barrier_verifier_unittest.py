"""Tests for barrier, policy, interval, verifier, and CLI components."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from barrier_tools.barriers.polynomial import PolynomialBarrier
from barrier_tools.dynamics.mountain_car import MountainCarContinuousDynamics
from barrier_tools.experiments.verify_mountain_car import run
from barrier_tools.policies.interval_bounds import bound_sequential
from barrier_tools.policies.pytorch_policy import ConstantPolicy, build_mlp
from barrier_tools.sets.boxes import Box
from barrier_tools.verification.branch_and_bound import (
    BarrierSpecification,
    BranchAndBoundVerifier,
    VerifierConfig,
)
from barrier_tools.verification.interval import cos_interval
from barrier_tools.verification.report import VerificationStatus


class IdentityDynamics:
    state_dim = 2
    action_dim = 1

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return state

    def interval_step(
        self,
        state_lower: torch.Tensor,
        state_upper: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del action_lower, action_upper
        return state_lower, state_upper


class LinearPositionBarrier:
    def value(self, state: torch.Tensor) -> torch.Tensor:
        return state[..., 0] + 1.0

    def interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return lower[..., 0] + 1.0, upper[..., 0] + 1.0

    def save(self, path: Path) -> None:
        torch.save({"type": "linear_position"}, path)


class BadBarrier:
    def value(self, state: torch.Tensor) -> torch.Tensor:
        return torch.ones(state.shape[:-1], dtype=state.dtype, device=state.device)

    def interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del upper
        value = torch.ones(lower.shape[:-1], dtype=lower.dtype, device=lower.device)
        return value, value

    def save(self, path: Path) -> None:
        torch.save({"type": "bad"}, path)


class BarrierVerifierTests(unittest.TestCase):
    def test_cos_interval_contains_samples(self) -> None:
        lower = torch.tensor([-1.0, 0.1, 2.0])
        upper = torch.tensor([1.0, 0.2, 7.0])
        cos_l, cos_u = cos_interval(lower, upper)
        for idx in range(lower.numel()):
            points = torch.linspace(lower[idx], upper[idx], 200)
            values = torch.cos(points)
            self.assertGreaterEqual(float(values.min().item()), float(cos_l[idx].item()) - 1e-6)
            self.assertLessEqual(float(values.max().item()), float(cos_u[idx].item()) + 1e-6)

    def test_policy_interval_contains_sampled_actions(self) -> None:
        torch.manual_seed(0)
        model = build_mlp([2, 4, 1], activation="tanh", final_tanh=True)
        lower = torch.tensor([[-0.5, -0.02]])
        upper = torch.tensor([[0.1, 0.04]])
        action_l, action_u = bound_sequential(model, lower, upper)
        samples = lower + torch.rand((256, 2)) * (upper - lower)
        actions = model(samples)
        self.assertTrue(bool((actions >= action_l - 1e-6).all().item()))
        self.assertTrue(bool((actions <= action_u + 1e-6).all().item()))

    def test_polynomial_interval_contains_sampled_values(self) -> None:
        barrier = PolynomialBarrier(degree=3, use_energy_base=False)
        with torch.no_grad():
            barrier.coefficients.copy_(torch.linspace(-0.2, 0.3, barrier.coefficients.numel()))
        lower = torch.tensor([[-0.8, -0.03]])
        upper = torch.tensor([[-0.2, 0.05]])
        h_l, h_u = barrier.interval(lower, upper)
        samples = lower + torch.rand((512, 2)) * (upper - lower)
        values = barrier.value(samples)
        self.assertGreaterEqual(float(values.min().item()), float(h_l.item()) - 1e-6)
        self.assertLessEqual(float(values.max().item()), float(h_u.item()) + 1e-6)

    def test_verifier_proves_simple_identity_case(self) -> None:
        spec = BarrierSpecification(
            initial_boxes=[Box(torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]))],
            unsafe_boxes=[Box(torch.tensor([-1.2, 0.0]), torch.tensor([-1.1, 0.0]))],
            invariant_boxes=[Box(torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]))],
            eps_init=1e-4,
            eps_unsafe=1e-4,
            eps_inv=1e-4,
            alpha=1.0,
        )
        verifier = BranchAndBoundVerifier(IdentityDynamics(), config=VerifierConfig(max_depth=4))
        report = verifier.verify(LinearPositionBarrier(), ConstantPolicy(0.0), spec)
        self.assertEqual(report.status, VerificationStatus.VERIFIED)
        self.assertGreater(report.margins["initial"], 0.0)

    def test_verifier_falsifies_unsafe_separation(self) -> None:
        spec = BarrierSpecification(
            initial_boxes=[Box(torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]))],
            unsafe_boxes=[Box(torch.tensor([-1.2, 0.0]), torch.tensor([-1.1, 0.0]))],
            invariant_boxes=[Box(torch.tensor([0.0, 0.0]), torch.tensor([0.1, 0.0]))],
            alpha=1.0,
        )
        verifier = BranchAndBoundVerifier(IdentityDynamics(), config=VerifierConfig(max_depth=4))
        report = verifier.verify(BadBarrier(), ConstantPolicy(0.0), spec)
        self.assertEqual(report.status, VerificationStatus.FALSIFIED)
        self.assertEqual(report.counterexample["category"], "unsafe")

    def test_unsafe_left_policy_is_falsified_for_position_barrier(self) -> None:
        spec = BarrierSpecification(
            initial_boxes=[Box(torch.tensor([-0.5, 0.0]), torch.tensor([-0.4, 0.0]))],
            unsafe_boxes=[Box(torch.tensor([-1.2, -0.07]), torch.tensor([-1.1, 0.07]))],
            invariant_boxes=[Box(torch.tensor([-1.0, -0.07]), torch.tensor([-0.99, -0.06]))],
            alpha=1.0,
            eps_inv=1e-6,
        )
        verifier = BranchAndBoundVerifier(
            MountainCarContinuousDynamics(),
            config=VerifierConfig(max_depth=8, max_boxes=1_000),
        )
        report = verifier.verify(LinearPositionBarrier(), ConstantPolicy(-1.0), spec)
        self.assertEqual(report.status, VerificationStatus.FALSIFIED)
        self.assertEqual(report.counterexample["category"], "invariance")

    def test_cli_smoke_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                f"""
run_id: smoke
seed: 3
output_dir: {tmpdir}
safety:
  p_safe: -1.0
  initial_boxes:
    - [[-0.6, -0.4], [0.0, 0.0]]
  unsafe_boxes:
    - [[-1.2, -1.0001], [-0.07, 0.07]]
  invariant_boxes:
    - [[-1.0, -0.98], [-0.07, -0.05]]
policy:
  type: constant
  action: -1.0
learner:
  epochs: 1
  samples_initial: 8
  samples_unsafe: 8
  samples_domain: 8
cegis:
  iterations: 1
verifier:
  max_depth: 2
  max_boxes: 32
  min_width: 0.001
rollouts:
  episodes: 1
""",
                encoding="utf-8",
            )
            summary = run(config_path)
            run_dir = Path(tmpdir) / "smoke"
            self.assertTrue((run_dir / "report.json").exists())
            self.assertTrue((run_dir / "barrier.pt").exists())
            self.assertIn(summary["report"]["status"], {"VERIFIED", "FALSIFIED", "UNKNOWN"})


if __name__ == "__main__":
    unittest.main()
