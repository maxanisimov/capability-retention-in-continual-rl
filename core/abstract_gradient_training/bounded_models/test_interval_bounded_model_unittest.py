"""Unit tests for IntervalBoundedModel activation support."""

from __future__ import annotations

import unittest

import torch

from abstract_gradient_training.bounded_models import CROWNBoundedModel
from abstract_gradient_training.bounded_models import IntervalBoundedModel


def _build_tanh_model(output_dim: int = 1) -> torch.nn.Sequential:
    torch.manual_seed(123)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, output_dim),
    ).double()


class IntervalBoundedModelTanhTests(unittest.TestCase):
    def test_constructor_and_bound_forward_accept_tanh(self) -> None:
        model = _build_tanh_model(output_dim=2)
        bounded_model = IntervalBoundedModel(model)
        batch = torch.randn(4, 3, dtype=torch.float64)

        lower, upper = bounded_model.bound_forward(batch, batch)

        self.assertEqual(lower.shape, (4, 2))
        self.assertEqual(upper.shape, (4, 2))

    def test_zero_width_tanh_bounds_match_pytorch_forward(self) -> None:
        model = _build_tanh_model(output_dim=2)
        bounded_model = IntervalBoundedModel(model)
        batch = torch.randn(6, 3, dtype=torch.float64)

        expected = model(batch)
        lower, upper = bounded_model.bound_forward(batch, batch)

        self.assertTrue(torch.allclose(lower, expected, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(upper, expected, atol=1e-10, rtol=1e-10))

    def test_nominal_tanh_backward_matches_autograd(self) -> None:
        model = _build_tanh_model()
        bounded_model = IntervalBoundedModel(model)
        batch = torch.randn(5, 3, dtype=torch.float64)
        targets = torch.randn(5, 1, dtype=torch.float64)

        expected_output = model(batch)
        loss = torch.nn.functional.mse_loss(expected_output, targets)
        expected_grads = torch.autograd.grad(loss, tuple(model.parameters()))

        custom_output = bounded_model.forward(batch, retain_intermediate=True)
        loss_grad = 2 * (custom_output - targets)
        custom_grads = bounded_model.backward(loss_grad)
        custom_grads = [grad.mean(dim=0) for grad in custom_grads]

        for expected, actual in zip(expected_grads, custom_grads):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-8, rtol=1e-8))

    def test_zero_width_tanh_bounded_backward_matches_autograd(self) -> None:
        model = _build_tanh_model()
        bounded_model = IntervalBoundedModel(model)
        batch = torch.randn(5, 3, dtype=torch.float64)
        targets = torch.randn(5, 1, dtype=torch.float64)

        expected_output = model(batch)
        loss = torch.nn.functional.mse_loss(expected_output, targets)
        expected_grads = torch.autograd.grad(loss, tuple(model.parameters()))

        lower, upper = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
        self.assertTrue(torch.allclose(lower, expected_output, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(upper, expected_output, atol=1e-10, rtol=1e-10))

        loss_grad = 2 * (expected_output.detach() - targets)
        grads_l, grads_u = bounded_model.bound_backward(loss_grad, loss_grad)
        grads_l = [grad.mean(dim=0) for grad in grads_l]
        grads_u = [grad.mean(dim=0) for grad in grads_u]

        for expected, lower_grad, upper_grad in zip(expected_grads, grads_l, grads_u):
            self.assertTrue(torch.allclose(lower_grad, expected, atol=1e-8, rtol=1e-8))
            self.assertTrue(torch.allclose(upper_grad, expected, atol=1e-8, rtol=1e-8))

    def test_tanh_bounds_enclose_sampled_outputs(self) -> None:
        model = _build_tanh_model(output_dim=2)
        bounded_model = IntervalBoundedModel(model)
        input_l = torch.tensor([[-1.0, -0.2, 0.3]], dtype=torch.float64)
        input_u = torch.tensor([[0.4, 0.7, 1.2]], dtype=torch.float64)

        output_l, output_u = bounded_model.bound_forward(input_l, input_u)
        generator = torch.Generator().manual_seed(456)
        samples = input_l + torch.rand((128, 3), generator=generator, dtype=torch.float64) * (
            input_u - input_l
        )
        sampled_outputs = model(samples)

        self.assertTrue(torch.all(sampled_outputs >= output_l - 1e-10).item())
        self.assertTrue(torch.all(sampled_outputs <= output_u + 1e-10).item())

    def test_crown_accepts_tanh(self) -> None:
        model = _build_tanh_model()
        bounded_model = CROWNBoundedModel(model, trainable=False)
        batch = torch.randn(3, 3, dtype=torch.float64)

        lower, upper = bounded_model.bound_forward(batch, batch)

        self.assertEqual(lower.shape, (3, 1))
        self.assertEqual(upper.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
