"""Unit tests for verified-margin constrained PPO helpers."""

from __future__ import annotations

import unittest

import torch

from experiments.utils.constrained_ppo import (
    VerifiedMarginConstraint,
    apply_safe_line_search,
    calibrate_margin_temperature,
)


class ConstantLogitActor(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits[: x.shape[0]].to(x.device)


def _payload() -> dict[str, torch.Tensor]:
    return {
        "state": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
        "actions": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    }


class VerifiedMarginConstrainedPPOTests(unittest.TestCase):
    def test_verified_margin_and_hard_accuracy_for_point_logits(self) -> None:
        actor = ConstantLogitActor(
            torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]], dtype=torch.float32),
        )
        constraint = VerifiedMarginConstraint.from_payload(
            _payload(),
            temperature=1,
            device="cpu",
        )

        self.assertGreater(constraint.margin(actor), 0.0)
        self.assertEqual(constraint.hard_accuracy(actor), 1.0)

    def test_temperature_calibration_selects_first_positive_margin(self) -> None:
        actor = ConstantLogitActor(
            torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]], dtype=torch.float32),
        )

        temperature, margin = calibrate_margin_temperature(
            actor,
            _payload(),
            inverse_temp_start=0,
            inverse_temp_max=3,
            device="cpu",
        )

        self.assertEqual(temperature, 1)
        self.assertGreater(margin, 0.0)

    def test_temperature_calibration_fails_when_no_temperature_is_safe(self) -> None:
        actor = ConstantLogitActor(
            torch.tensor([[0.0, 5.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )

        with self.assertRaisesRegex(ValueError, "Could not calibrate"):
            calibrate_margin_temperature(
                actor,
                _payload(),
                inverse_temp_start=0,
                inverse_temp_max=3,
                device="cpu",
            )

    def test_safe_line_search_backtracks_to_first_safe_alpha(self) -> None:
        actor = torch.nn.Sequential(torch.nn.Linear(3, 4))
        layer = actor[0]
        assert isinstance(layer, torch.nn.Linear)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0]))
        old_params = [param.detach().clone() for param in actor.parameters()]
        candidate_params = [param.detach().clone() for param in actor.parameters()]
        candidate_params[1] = torch.tensor([0.0, 5.0, 0.0, 0.0])
        with torch.no_grad():
            layer.bias.copy_(candidate_params[1])

        constraint = VerifiedMarginConstraint.from_payload(
            {
                "state": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                "actions": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            },
            temperature=1,
            device="cpu",
        )
        decision = apply_safe_line_search(
            actor,
            old_actor_params=old_params,
            candidate_actor_params=candidate_params,
            constraint=constraint,
            max_backtracks=2,
            backtrack_coef=0.5,
        )

        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(decision.alpha, 0.25)
        self.assertGreater(decision.margin, 0.0)

    def test_safe_line_search_restores_old_params_when_all_alphas_fail(self) -> None:
        actor = torch.nn.Sequential(torch.nn.Linear(3, 4))
        layer = actor[0]
        assert isinstance(layer, torch.nn.Linear)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0]))
        old_params = [param.detach().clone() for param in actor.parameters()]
        candidate_params = [param.detach().clone() for param in actor.parameters()]
        candidate_params[1] = torch.tensor([0.0, 5.0, 0.0, 0.0])
        with torch.no_grad():
            layer.bias.copy_(candidate_params[1])

        constraint = VerifiedMarginConstraint.from_payload(
            {
                "state": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                "actions": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            },
            temperature=1,
            device="cpu",
        )
        decision = apply_safe_line_search(
            actor,
            old_actor_params=old_params,
            candidate_actor_params=candidate_params,
            constraint=constraint,
            max_backtracks=1,
            backtrack_coef=0.5,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.alpha, 0.0)
        torch.testing.assert_close(layer.bias.detach(), old_params[1])


if __name__ == "__main__":
    unittest.main()
