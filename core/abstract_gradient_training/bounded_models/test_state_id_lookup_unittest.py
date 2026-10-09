"""Tests for exact one-hot semantics without dense one-hot tensors."""

from __future__ import annotations

import unittest

import torch
from src.verification.api import build_bounded_model
from src.verification.compatibility import UnsupportedLayerError

from abstract_gradient_training.bounded_models import (
    IntervalBoundedModel,
    StateIdLookupLinear,
)


class StateIdLookupLinearTests(unittest.TestCase):
    def _models(self) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
        torch.manual_seed(11)
        dense = torch.nn.Sequential(
            torch.nn.Linear(7, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 3),
        )
        lookup = torch.nn.Sequential(
            StateIdLookupLinear.from_linear(dense[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 3),
        )
        lookup[2].load_state_dict(dense[2].state_dict())
        return dense, lookup

    def test_nominal_forward_equals_dense_one_hot(self) -> None:
        dense, lookup = self._models()
        ids = torch.tensor([[0], [3], [6], [3]], dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(ids[:, 0], 7).float()
        torch.testing.assert_close(lookup(ids), dense(one_hot), rtol=0, atol=0)

    def test_ibp_bounds_and_bound_gradients_equal_dense_one_hot(self) -> None:
        dense, lookup = self._models()
        ids = torch.tensor([[0], [3], [6], [3]], dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(ids[:, 0], 7).float()
        dense_bounded = IntervalBoundedModel(dense)
        lookup_bounded = IntervalBoundedModel(lookup)
        for bounded in (dense_bounded, lookup_bounded):
            for lower, upper in zip(bounded.param_l, bounded.param_u):
                lower.data.sub_(0.01)
                upper.data.add_(0.02)
                lower.requires_grad_(True)
                upper.requires_grad_(True)

        dense_l, dense_u = dense_bounded.bound_forward(one_hot, one_hot)
        lookup_l, lookup_u = lookup_bounded.bound_forward(ids, ids)
        torch.testing.assert_close(lookup_l, dense_l, rtol=0, atol=0)
        torch.testing.assert_close(lookup_u, dense_u, rtol=0, atol=0)

        (dense_l.sum() + dense_u.sum()).backward()
        (lookup_l.sum() + lookup_u.sum()).backward()
        for dense_param, lookup_param in zip(
            [*dense_bounded.param_l, *dense_bounded.param_u],
            [*lookup_bounded.param_l, *lookup_bounded.param_u],
        ):
            torch.testing.assert_close(
                lookup_param.grad, dense_param.grad, rtol=0, atol=0
            )

    def test_rejects_invalid_and_interval_state_ids(self) -> None:
        layer = StateIdLookupLinear(4, 2)
        with self.assertRaises(TypeError):
            layer(torch.tensor([[1.0]]))
        with self.assertRaises(IndexError):
            layer(torch.tensor([[4]]))
        bounded = IntervalBoundedModel(torch.nn.Sequential(layer))
        with self.assertRaisesRegex(ValueError, "point state IDs only"):
            bounded.bound_forward(torch.tensor([[1]]), torch.tensor([[2]]))

    def test_lookup_is_ibp_only(self) -> None:
        model = torch.nn.Sequential(StateIdLookupLinear(4, 2))
        self.assertIsInstance(build_bounded_model(model, "IBP"), IntervalBoundedModel)
        with self.assertRaises(UnsupportedLayerError):
            build_bounded_model(model, "CROWN")


if __name__ == "__main__":
    unittest.main()
