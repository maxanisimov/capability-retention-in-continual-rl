"""Tests for adaptive PSPO certificates over continuous input boxes."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import gymnasium as gym
import torch
from continuous_state_shields import MountainCarShield
from torch.utils.data import TensorDataset

from provably_safe_policy_optimisation import AdaptiveSafePPO
from provably_safe_policy_optimisation import adaptive_safe_ppo as adaptive_module
from provably_safe_policy_optimisation.adaptive_safe_ppo import (
    validate_interval_certificate_dataset,
)


def _certificate() -> TensorDataset:
    return TensorDataset(
        torch.tensor([[-1.2, -0.07]]),
        torch.tensor([[-1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )


def _base(action: int = 2) -> dict[str, torch.Tensor]:
    bias = torch.zeros(3)
    bias[action] = 5.0
    return {
        "action_net.weight": torch.zeros((3, 2)),
        "action_net.bias": bias,
    }


class ContinuousAdaptiveSafePPOTests(unittest.TestCase):
    def _make(self, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("base_policy_state_dict", _base())
        kwargs.setdefault("interval_certificate_dataset", _certificate())
        kwargs.setdefault("directional_rashomon_growth", False)
        kwargs.setdefault("stop_when_proposal_contained", False)
        model = AdaptiveSafePPO(
            "MlpPolicy",
            gym.make("MountainCar-v0"),
            shield=MountainCarShield(),
            policy_kwargs={"net_arch": []},
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=0,
            device="cpu",
            verbose=0,
            **kwargs,
        )
        self.addCleanup(model.get_env().close)
        return model

    def test_complete_interval_is_certified_and_reported(self) -> None:
        model = self._make()
        self.assertEqual(model._greedy_safe_rate_now(), 1.0)
        diagnostics = model.adaptive_diagnostics()
        self.assertEqual(diagnostics["certificate_input_mode"], "intervals")
        self.assertEqual(diagnostics["certificate_regions"], 1)

    def test_unsafe_base_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "hard shield invariant"):
            self._make(base_policy_state_dict=_base(action=0))

    def test_interval_certificates_reject_zonotope_regions(self) -> None:
        with self.assertRaisesRegex(ValueError, "orthotope"):
            self._make(
                safe_region_shape="zonotope",
                directional_rashomon_growth=False,
                stop_when_proposal_contained=False,
            )

    def test_rashomon_engine_receives_interval_dataset_flag(self) -> None:
        calls: list[dict[str, object]] = []
        original = adaptive_module._run_rashomon_engine

        def stub(model, dataset, **kwargs):  # type: ignore[no-untyped-def]
            calls.append({"dataset": dataset, **kwargs})
            params = [parameter.detach().clone() for parameter in model.parameters()]
            bounded = SimpleNamespace(param_l=params, param_u=params)
            return SimpleNamespace(
                bounded_models=[bounded],
                certificates=[[SimpleNamespace(min_hard_acc=1.0)]],
                iterations_run=1,
            )

        adaptive_module._run_rashomon_engine = stub
        self.addCleanup(setattr, adaptive_module, "_run_rashomon_engine", original)
        model = self._make(rashomon_n_iters=1, rashomon_checkpoint=1)
        selected = model._compute_rashomon_around_last_safe()
        self.assertIsNotNone(selected)
        self.assertTrue(calls[0]["has_input_intervals"])
        self.assertEqual(len(calls[0]["dataset"].tensors), 3)  # type: ignore[union-attr]
        self.assertIsNone(calls[0]["inverse_temp"])

    def test_segment_engine_receives_interval_dataset_flag(self) -> None:
        """A segment safe region is allowed with interval certificates.

        The segment engine reuses the point-input code path by wrapping every
        batch in an ``IntervalTensor``, so the only thing the PSPO layer has to
        get right is forwarding the flag and the proposal direction.
        """
        calls: list[dict[str, object]] = []
        original = adaptive_module._run_segment_engine

        def stub(model, dataset, **kwargs):  # type: ignore[no-untyped-def]
            calls.append({"dataset": dataset, **kwargs})
            params = [parameter.detach().clone() for parameter in model.parameters()]
            return SimpleNamespace(
                regions=[],
                certificates=[],
                alpha=0.0,
                splits_used=1,
                iterations_run=1,
                target_contained_and_certified=False,
                _params=params,
            )

        adaptive_module._run_segment_engine = stub
        self.addCleanup(setattr, adaptive_module, "_run_segment_engine", original)
        model = self._make(safe_region_shape="segment")
        proposal = [
            parameter.detach().clone() + 1e-3 for parameter in model._live_actor_params
        ]
        model._compute_rashomon_around_last_safe(stop_target_params=proposal)
        self.assertTrue(calls[0]["has_input_intervals"])
        self.assertEqual(len(calls[0]["dataset"].tensors), 3)  # type: ignore[union-attr]
        self.assertEqual(
            len(calls[0]["delta"]),  # type: ignore[arg-type]
            len(model._live_actor_params),
        )

    def test_segment_requires_the_proposal_direction(self) -> None:
        model = self._make(safe_region_shape="segment")
        with self.assertRaisesRegex(ValueError, "stop_target_params"):
            model._compute_rashomon_around_last_safe()


class IntervalDatasetValidationTests(unittest.TestCase):
    def test_rejects_invalid_box_and_action_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation space"):
            validate_interval_certificate_dataset(
                TensorDataset(
                    torch.zeros((1, 3)),
                    torch.ones((1, 3)),
                    torch.ones((1, 3)),
                ),
                observation_shape=(2,),
                n_actions=3,
            )
        with self.assertRaisesRegex(ValueError, "n_boxes, n_actions"):
            validate_interval_certificate_dataset(
                TensorDataset(
                    torch.zeros((1, 2)),
                    torch.ones((1, 2)),
                    torch.ones((1, 2)),
                ),
                observation_shape=(2,),
                n_actions=3,
            )


if __name__ == "__main__":
    unittest.main()
