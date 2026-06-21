"""Unit tests for tanh support in interval and zonotope verification."""

from __future__ import annotations

import copy
import unittest

import torch

from src.IntervalTensor import IntervalTensor
from src.verification.verify import bound_forward_pass
from src.verification.zonotope_tensor import (
    ZonotopeTensor,
    tanh_affine_residual_bounds,
)


SCALAR_INTERVALS = [
    (-3.0, -1.0),
    (-1.0, -0.1),
    (0.1, 1.0),
    (1.0, 3.0),
    (-1.0, 1.0),
    (-3.0, 0.2),
    (-0.2, 3.0),
    (0.5, 0.5),
    (-0.5, -0.5),
    (0.5, 0.500000001),
    (-0.500000001, -0.5),
    (-20.0, 20.0),
    (20.0, 100.0),
    (-100.0, -20.0),
]


def _build_tanh_model() -> torch.nn.Sequential:
    torch.manual_seed(123)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 2),
    ).double()


def _perturbed_model(
    model: torch.nn.Sequential, flat_delta: torch.Tensor
) -> torch.nn.Sequential:
    sampled_model = copy.deepcopy(model)
    offset = 0
    with torch.no_grad():
        for param in sampled_model.parameters():
            n_param = param.numel()
            delta = flat_delta[offset : offset + n_param].reshape_as(param)
            param.copy_(param + delta)
            offset += n_param
    return sampled_model


class ZonotopeTanhTests(unittest.TestCase):
    def test_scalar_tanh_affine_residual_bounds_are_sound(self) -> None:
        for lower, upper in SCALAR_INTERVALS:
            with self.subTest(interval=(lower, upper)):
                l = torch.tensor(lower, dtype=torch.float64)
                u = torch.tensor(upper, dtype=torch.float64)
                slope, residual_l, residual_u = tanh_affine_residual_bounds(l, u)

                for coeff in (slope, residual_l, residual_u):
                    self.assertTrue(torch.isfinite(coeff).all().item())

                z = torch.linspace(lower, upper, steps=4097, dtype=torch.float64)
                tanh_z = torch.tanh(z)
                approx_l = slope * z + residual_l
                approx_u = slope * z + residual_u

                self.assertTrue(torch.all(approx_l <= tanh_z + 1e-6).item())
                self.assertTrue(torch.all(tanh_z <= approx_u + 1e-6).item())

    def test_tanh_affine_residual_bounds_are_vectorized(self) -> None:
        l = torch.tensor(
            [[-3.0, 0.1, -1.0], [0.5, 20.0, -100.0]],
            dtype=torch.float64,
        )
        u = torch.tensor(
            [[-1.0, 1.0, 1.0], [0.5, 100.0, -20.0]],
            dtype=torch.float64,
        )

        slope, residual_l, residual_u = tanh_affine_residual_bounds(l, u)

        for coeff in (slope, residual_l, residual_u):
            self.assertEqual(coeff.shape, l.shape)
            self.assertEqual(coeff.dtype, l.dtype)
            self.assertEqual(coeff.device, l.device)
            self.assertTrue(torch.isfinite(coeff).all().item())

        grid = torch.linspace(0.0, 1.0, steps=1025, dtype=torch.float64)
        z = l.unsqueeze(0) + grid.reshape(-1, 1, 1) * (u - l).unsqueeze(0)
        tanh_z = torch.tanh(z)
        approx_l = slope.unsqueeze(0) * z + residual_l.unsqueeze(0)
        approx_u = slope.unsqueeze(0) * z + residual_u.unsqueeze(0)

        self.assertTrue(torch.all(approx_l <= tanh_z + 1e-6).item())
        self.assertTrue(torch.all(tanh_z <= approx_u + 1e-6).item())

    def test_interval_tanh_is_monotone(self) -> None:
        interval = IntervalTensor(
            torch.tensor([[-2.0, -0.5, 0.1]], dtype=torch.float64),
            torch.tensor([[0.0, 1.5, 3.0]], dtype=torch.float64),
        )

        output = interval.tanh()

        self.assertTrue(torch.equal(output.lb, torch.tanh(interval.lb)))
        self.assertTrue(torch.equal(output.ub, torch.tanh(interval.ub)))

    def test_zero_width_zonotope_tanh_matches_torch(self) -> None:
        center = torch.tensor(
            [[-1.0, 0.0, 0.7], [2.0, -3.0, 0.2]],
            dtype=torch.float64,
        )
        generators = torch.randn(1, *center.shape, dtype=torch.float64)
        coefficients = IntervalTensor(torch.zeros(1, dtype=torch.float64))
        zonotope = ZonotopeTensor(center, generators, coefficients)

        output = zonotope.tanh()
        output_interval = output.concretize()
        expected = torch.tanh(center)

        self.assertEqual(output.shape, center.shape)
        self.assertEqual(output.center.lb.dtype, center.dtype)
        self.assertEqual(output.center.lb.device, center.device)
        self.assertTrue(torch.allclose(output_interval.lb, expected, atol=1e-10))
        self.assertTrue(torch.allclose(output_interval.ub, expected, atol=1e-10))

    def test_zonotope_tanh_encloses_sampled_points(self) -> None:
        torch.manual_seed(456)
        center = torch.randn(2, 3, dtype=torch.float64)
        generators = 0.2 * torch.randn(4, *center.shape, dtype=torch.float64)
        coefficients = IntervalTensor(
            -torch.ones(4, dtype=torch.float64),
            torch.ones(4, dtype=torch.float64),
        )
        zonotope = ZonotopeTensor(center, generators, coefficients)

        output_interval = zonotope.tanh().concretize()
        generator = torch.Generator().manual_seed(789)
        coeff_samples = 2 * torch.rand(
            (256, 4), generator=generator, dtype=torch.float64
        ) - 1

        for coeff_sample in coeff_samples:
            concrete = center + (
                coeff_sample.reshape(-1, 1, 1) * generators
            ).sum(dim=0)
            expected = torch.tanh(concrete)
            self.assertTrue(torch.all(expected >= output_interval.lb - 1e-6).item())
            self.assertTrue(torch.all(expected <= output_interval.ub + 1e-6).item())

    def test_bound_forward_pass_accepts_tanh_and_encloses_samples(self) -> None:
        model = _build_tanh_model()
        inputs = torch.randn(6, 3, dtype=torch.float64)
        n_params = sum(param.numel() for param in model.parameters())
        generator = torch.Generator().manual_seed(321)
        flat_generators = 0.03 * torch.randn(
            4, n_params, generator=generator, dtype=torch.float64
        )
        coefficients = IntervalTensor(
            -torch.ones(4, dtype=torch.float64),
            torch.ones(4, dtype=torch.float64),
        )

        bounds = [
            bound_forward_pass(
                model,
                flat_generators,
                coefficients,
                inputs,
                use_zonotopes=use_zonotopes,
            )
            for use_zonotopes in (True, False)
        ]
        coeff_samples = 2 * torch.rand(
            (128, 4), generator=generator, dtype=torch.float64
        ) - 1

        for coeff_sample in coeff_samples:
            flat_delta = coeff_sample @ flat_generators
            sampled_model = _perturbed_model(model, flat_delta)
            expected = sampled_model(inputs)
            for output_interval in bounds:
                self.assertTrue(torch.all(expected >= output_interval.lb - 1e-6).item())
                self.assertTrue(torch.all(expected <= output_interval.ub + 1e-6).item())


if __name__ == "__main__":
    unittest.main()
