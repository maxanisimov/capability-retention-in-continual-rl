"""Unit tests for CROWN tanh relaxations."""

from __future__ import annotations

import unittest

import torch

from abstract_gradient_training.bounded_models import CROWNBoundedModel
from abstract_gradient_training.bounded_models._crown_bounds import (
    InputNode,
    TanhNode,
    tanh_linear_bounds,
)


SCALAR_INTERVALS = [
    (-3.0, -1.0),
    (-1.0, -0.1),
    (0.1, 1.0),
    (1.0, 3.0),
    (0.5, 0.5),
    (-0.5, -0.5),
    (0.5, 0.500000001),
    (-0.500000001, -0.5),
    (-1.0, 1.0),
    (-3.0, 0.2),
    (-0.2, 3.0),
    (-20.0, 20.0),
    (20.0, 100.0),
    (-100.0, -20.0),
]


def _build_tanh_model(output_dim: int = 2) -> torch.nn.Sequential:
    torch.manual_seed(321)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 6),
        torch.nn.Tanh(),
        torch.nn.Linear(6, output_dim),
    ).double()


def _build_relu_tanh_model() -> torch.nn.Sequential:
    torch.manual_seed(987)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2),
    ).double()


def _assert_tanh_node_envelope_sound(
    test_case: unittest.TestCase,
    node: TanhNode,
    lower: float,
    upper: float,
) -> None:
    node.clear_cached()
    node.update_relaxation()
    z = torch.linspace(lower, upper, steps=4097, dtype=torch.float64)
    tanh_z = torch.tanh(z)
    y_l = node.alpha_l.squeeze() * z + node.beta_l.squeeze()
    y_u = node.alpha_u.squeeze() * z + node.beta_u.squeeze()

    test_case.assertTrue(torch.all(y_l <= tanh_z + 1e-6).item())
    test_case.assertTrue(torch.all(tanh_z <= y_u + 1e-6).item())


class CrownTanhRelaxationTests(unittest.TestCase):
    def test_scalar_tanh_linear_bounds_are_sound(self) -> None:
        for lower, upper in SCALAR_INTERVALS:
            with self.subTest(interval=(lower, upper)):
                l = torch.tensor(lower, dtype=torch.float64)
                u = torch.tensor(upper, dtype=torch.float64)
                a_l, b_l, a_u, b_u = tanh_linear_bounds(l, u)

                for coeff in (a_l, b_l, a_u, b_u):
                    self.assertTrue(torch.isfinite(coeff).all().item())

                z = torch.linspace(lower, upper, steps=4097, dtype=torch.float64)
                tanh_z = torch.tanh(z)
                y_l = a_l * z + b_l
                y_u = a_u * z + b_u

                self.assertTrue(torch.all(y_l <= tanh_z + 1e-6).item())
                self.assertTrue(torch.all(tanh_z <= y_u + 1e-6).item())

    def test_tanh_linear_bounds_support_batched_tensors(self) -> None:
        l = torch.tensor(
            [[-3.0, 0.1, 0.5], [-1.0, 1.0, -20.0]], dtype=torch.float64
        )
        u = torch.tensor(
            [[-1.0, 1.0, 0.5], [1.0, 3.0, 20.0]], dtype=torch.float64
        )

        bounds = tanh_linear_bounds(l, u)

        for coeff in bounds:
            self.assertEqual(coeff.shape, l.shape)
            self.assertEqual(coeff.dtype, l.dtype)
            self.assertEqual(coeff.device, l.device)
            self.assertTrue(torch.isfinite(coeff).all().item())

        grid = torch.linspace(0.0, 1.0, steps=1025, dtype=torch.float64)
        z = l.unsqueeze(0) + grid.reshape(-1, 1, 1) * (u - l).unsqueeze(0)
        tanh_z = torch.tanh(z)
        a_l, b_l, a_u, b_u = bounds
        y_l = a_l.unsqueeze(0) * z + b_l.unsqueeze(0)
        y_u = a_u.unsqueeze(0) * z + b_u.unsqueeze(0)

        self.assertTrue(torch.all(y_l <= tanh_z + 1e-6).item())
        self.assertTrue(torch.all(tanh_z <= y_u + 1e-6).item())

    def test_crown_accepts_tanh_and_matches_zero_width_forward(self) -> None:
        model = _build_tanh_model()
        bounded_model = CROWNBoundedModel(model, trainable=False)
        batch = torch.randn(5, 3, dtype=torch.float64)

        expected = model(batch)
        lower, upper = bounded_model.bound_forward(batch, batch)

        self.assertTrue(torch.allclose(lower, expected, atol=1e-8, rtol=1e-8))
        self.assertTrue(torch.allclose(upper, expected, atol=1e-8, rtol=1e-8))

    def test_crown_tanh_bounds_enclose_sampled_outputs(self) -> None:
        model = _build_tanh_model()
        bounded_model = CROWNBoundedModel(model, trainable=False)
        input_l = torch.tensor([[-1.0, -0.3, 0.2]], dtype=torch.float64)
        input_u = torch.tensor([[0.4, 0.9, 1.3]], dtype=torch.float64)

        output_l, output_u = bounded_model.bound_forward(input_l, input_u)
        generator = torch.Generator().manual_seed(654)
        samples = input_l + torch.rand((256, 3), generator=generator, dtype=torch.float64) * (
            input_u - input_l
        )
        sampled_outputs = model(samples)

        self.assertTrue(torch.all(sampled_outputs >= output_l - 1e-6).item())
        self.assertTrue(torch.all(sampled_outputs <= output_u + 1e-6).item())

    def test_linear_gradient_mode_handles_tanh(self) -> None:
        model = _build_tanh_model(output_dim=1)
        bounded_model = CROWNBoundedModel(
            model,
            trainable=True,
            gradient_bound_mode="linear",
        )
        batch = torch.randn(4, 3, dtype=torch.float64)

        lower, upper = bounded_model.bound_forward(batch, batch, retain_intermediate=True)
        grads_l, grads_u = bounded_model.bound_backward(
            torch.ones_like(lower),
            torch.ones_like(upper),
        )

        self.assertEqual(len(grads_l), len(tuple(model.parameters())))
        self.assertEqual(len(grads_u), len(tuple(model.parameters())))
        for grad_l, grad_u in zip(grads_l, grads_u):
            self.assertTrue(torch.isfinite(grad_l).all().item())
            self.assertTrue(torch.isfinite(grad_u).all().item())

    def test_tanh_node_fixed_mode_has_no_optimizable_parameters(self) -> None:
        input_node = InputNode(
            torch.tensor([[-1.0, 0.2]], dtype=torch.float64),
            torch.tensor([[0.5, 1.4]], dtype=torch.float64),
        )
        node = TanhNode(input_node)

        self.assertEqual(node.optimizable_parameters(), [])

    def test_tanh_node_optimizable_mode_exposes_parameters(self) -> None:
        input_node = InputNode(
            torch.tensor([[-1.0, 0.2]], dtype=torch.float64),
            torch.tensor([[0.5, 1.4]], dtype=torch.float64),
        )
        node = TanhNode(input_node, tanh_relaxation="optimizable")
        params = node.optimizable_parameters()

        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, (1, 2))
        self.assertEqual(params[1].shape, (1, 2))

    def test_optimized_tanh_node_envelopes_remain_sound(self) -> None:
        for lower, upper in SCALAR_INTERVALS:
            with self.subTest(interval=(lower, upper)):
                input_node = InputNode(
                    torch.tensor([[lower]], dtype=torch.float64),
                    torch.tensor([[upper]], dtype=torch.float64),
                )
                node = TanhNode(input_node, tanh_relaxation="optimizable")
                optimizer = torch.optim.Adam(node.optimizable_parameters(), lr=0.1)

                for _ in range(5):
                    optimizer.zero_grad()
                    node.clear_cached()
                    out_l, out_u = node.concretize()
                    loss = (out_u - out_l).sum()
                    loss.backward()
                    optimizer.step()

                _assert_tanh_node_envelope_sound(self, node, lower, upper)

    def test_optimized_tanh_node_parameters_receive_gradients(self) -> None:
        input_node = InputNode(
            torch.tensor([[-1.0, -0.2]], dtype=torch.float64),
            torch.tensor([[0.3, 1.2]], dtype=torch.float64),
        )
        node = TanhNode(input_node, tanh_relaxation="optimizable")
        out_l, out_u = node.concretize()
        loss = (out_u - out_l).sum()
        loss.backward()

        for param in node.optimizable_parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all().item())

    def test_crown_optimized_tanh_matches_zero_width_forward(self) -> None:
        model = _build_tanh_model()
        bounded_model = CROWNBoundedModel(
            model,
            trainable=False,
            tanh_relaxation="optimizable",
            alpha_crown_iters=5,
        )
        batch = torch.randn(5, 3, dtype=torch.float64)

        expected = model(batch)
        lower, upper = bounded_model.bound_forward(batch, batch)

        self.assertTrue(torch.allclose(lower, expected, atol=1e-8, rtol=1e-8))
        self.assertTrue(torch.allclose(upper, expected, atol=1e-8, rtol=1e-8))

    def test_crown_optimized_tanh_bounds_enclose_sampled_outputs(self) -> None:
        model = _build_tanh_model()
        bounded_model = CROWNBoundedModel(
            model,
            trainable=False,
            tanh_relaxation="optimizable",
            alpha_crown_iters=5,
        )
        input_l = torch.tensor([[-1.0, -0.3, 0.2]], dtype=torch.float64)
        input_u = torch.tensor([[0.4, 0.9, 1.3]], dtype=torch.float64)

        output_l, output_u = bounded_model.bound_forward(input_l, input_u)
        generator = torch.Generator().manual_seed(321)
        samples = input_l + torch.rand((256, 3), generator=generator, dtype=torch.float64) * (
            input_u - input_l
        )
        sampled_outputs = model(samples)

        self.assertTrue(torch.all(sampled_outputs >= output_l - 1e-6).item())
        self.assertTrue(torch.all(sampled_outputs <= output_u + 1e-6).item())

    def test_crown_optimized_mixed_relu_tanh_bounds_are_sound(self) -> None:
        model = _build_relu_tanh_model()
        bounded_model = CROWNBoundedModel(
            model,
            trainable=False,
            relu_relaxation="optimizable",
            tanh_relaxation="optimizable",
            alpha_crown_iters=5,
        )
        input_l = torch.tensor([[-0.8, -0.5, 0.1]], dtype=torch.float64)
        input_u = torch.tensor([[0.9, 0.6, 1.0]], dtype=torch.float64)

        output_l, output_u = bounded_model.bound_forward(input_l, input_u)
        generator = torch.Generator().manual_seed(654)
        samples = input_l + torch.rand((256, 3), generator=generator, dtype=torch.float64) * (
            input_u - input_l
        )
        sampled_outputs = model(samples)

        self.assertTrue(torch.all(sampled_outputs >= output_l - 1e-6).item())
        self.assertTrue(torch.all(sampled_outputs <= output_u + 1e-6).item())


if __name__ == "__main__":
    unittest.main()
