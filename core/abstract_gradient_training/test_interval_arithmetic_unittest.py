"""Tests for the interval matmul fast paths: the one-hot row lookup and the
point-operand short-circuit in Rump's algorithm, plus the unchanged behaviour of
the general path."""

import unittest
from unittest import mock

import torch

from abstract_gradient_training import interval_arithmetic as ia
from abstract_gradient_training.bounded_models import IntervalBoundedModel


def _reference_rump(
    A_l: torch.Tensor, A_u: torch.Tensor, B_l: torch.Tensor, B_u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unspecialised form of Rump's algorithm, as a reference."""
    A_mu, A_r = (A_u + A_l) / 2, (A_u - A_l) / 2
    B_mu, B_r = (B_u + B_l) / 2, (B_u - B_l) / 2
    H_mu = A_mu @ B_mu
    H_r = torch.abs(A_mu) @ B_r + A_r @ torch.abs(B_mu) + A_r @ B_r
    return H_mu - H_r, H_mu + H_r


def _one_hot(indices: list[int], width: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(indices), width).double()


def _random_weight_interval(
    n: int, p: int, radius: float = 0.1, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    B_mu = torch.randn(n, p, generator=generator, dtype=torch.float64)
    B_r = radius * torch.rand(n, p, generator=generator, dtype=torch.float64)
    return B_mu - B_r, B_mu + B_r


class TestOneHotRowLookup(unittest.TestCase):
    """A point matrix of one-hot rows must be handled as an exact row lookup."""

    def test_matches_dense_matmul_for_every_method(self):
        A = _one_hot([3, 0, 7, 3], width=8)
        B_l, B_u = _random_weight_interval(8, 5)
        for method in ("rump", "exact", "nguyen"):
            with self.subTest(method=method):
                H_l, H_u = ia.propagate_matmul(A, A.clone(), B_l, B_u, method)
                self.assertTrue(torch.equal(H_l, B_l[[3, 0, 7, 3]]))
                self.assertTrue(torch.equal(H_u, B_u[[3, 0, 7, 3]]))

    def test_agrees_with_the_dense_computation_to_within_rounding(self):
        # The lookup is the exact image of the one-hot rows, while the dense path
        # reaches it through a midpoint/radius round trip, so the two agree only
        # up to the rounding of (l + u) / 2 -/+ (u - l) / 2.
        A = _one_hot([2, 9, 9, 0, 4], width=12)
        B_l, B_u = _random_weight_interval(12, 6, seed=1)
        ref_l, ref_u = _reference_rump(A, A.clone(), B_l, B_u)
        H_l, H_u = ia.propagate_matmul(A, A.clone(), B_l, B_u, "rump")
        self.assertTrue(torch.allclose(H_l, ref_l, rtol=0.0, atol=1e-15))
        self.assertTrue(torch.allclose(H_u, ref_u, rtol=0.0, atol=1e-15))

    def test_is_the_exact_image_of_the_selected_rows(self):
        A = _one_hot([2, 9, 9, 0, 4], width=12)
        B_l, B_u = _random_weight_interval(12, 6, seed=1)
        H_l, H_u = ia.propagate_matmul(A, A.clone(), B_l, B_u, "rump")
        self.assertTrue(torch.equal(H_l, A @ B_l))
        self.assertTrue(torch.equal(H_u, A @ B_u))

    def test_fast_path_is_actually_taken(self):
        A = _one_hot([1, 0], width=4)
        B_l, B_u = _random_weight_interval(4, 3, seed=2)
        with mock.patch.object(
            ia, "propagate_matmul_rump", side_effect=AssertionError("dense path used")
        ):
            H_l, H_u = ia.propagate_matmul(A, A.clone(), B_l, B_u, "rump")
        self.assertTrue(torch.equal(H_l, B_l[[1, 0]]))
        self.assertTrue(torch.equal(H_u, B_u[[1, 0]]))

    def test_result_does_not_alias_the_weight_bounds(self):
        A = _one_hot([0, 1], width=2)
        B_l, B_u = _random_weight_interval(2, 3, seed=3)
        H_l, _ = ia.propagate_matmul(A, A.clone(), B_l, B_u, "rump")
        H_l += 1.0
        self.assertTrue(torch.equal(B_l, _random_weight_interval(2, 3, seed=3)[0]))

    def test_propagate_affine_adds_the_bias_to_the_lookup(self):
        # propagate_affine computes W @ x, so the selector is the first operand.
        selector = _one_hot([2, 0], width=3)
        x_l, x_u = _random_weight_interval(3, 4, seed=4)
        b_l = torch.arange(4, dtype=torch.float64)
        b_u = b_l + 0.5
        H_l, H_u = ia.propagate_affine(x_l, x_u, selector, selector.clone(), b_l, b_u)
        self.assertTrue(torch.equal(H_l, x_l[[2, 0]] + b_l))
        self.assertTrue(torch.equal(H_u, x_u[[2, 0]] + b_u))


class TestOneHotDetection(unittest.TestCase):
    """The lookup must be declined whenever A is not a point one-hot matrix."""

    def setUp(self):
        self.B_l, self.B_u = _random_weight_interval(4, 3, seed=5)

    def _assert_declined(self, A_l: torch.Tensor, A_u: torch.Tensor):
        self.assertIsNone(ia._point_onehot_row_indices(A_l, A_u, self.B_l))
        ref_l, ref_u = _reference_rump(A_l, A_u, self.B_l, self.B_u)
        H_l, H_u = ia.propagate_matmul(A_l, A_u, self.B_l, self.B_u, "rump")
        self.assertTrue(torch.equal(H_l, ref_l))
        self.assertTrue(torch.equal(H_u, ref_u))

    def test_declines_proper_interval_input(self):
        A = _one_hot([1, 2], width=4)
        self._assert_declined(A - 0.01, A + 0.01)

    def test_declines_two_non_zeros_in_a_row(self):
        A = _one_hot([1, 2], width=4)
        A[0, 3] = 1.0
        self._assert_declined(A, A.clone())

    def test_declines_an_all_zero_row(self):
        A = _one_hot([1, 2], width=4)
        A[1] = 0.0
        self._assert_declined(A, A.clone())

    def test_declines_a_non_unit_entry(self):
        A = 2.0 * _one_hot([1, 2], width=4)
        self._assert_declined(A, A.clone())

    def test_declines_a_negative_unit_entry(self):
        A = -1.0 * _one_hot([1, 2], width=4)
        self._assert_declined(A, A.clone())

    def test_declines_non_matrix_operands(self):
        A = _one_hot([1, 2], width=4).unsqueeze(0)
        self.assertIsNone(ia._point_onehot_row_indices(A, A.clone(), self.B_l))
        self.assertIsNone(
            ia._point_onehot_row_indices(
                A.squeeze(0), A.squeeze(0), self.B_l.unsqueeze(0)
            )
        )

    def test_declines_mismatched_inner_dimension(self):
        A = _one_hot([1, 2], width=5)
        self.assertIsNone(ia._point_onehot_row_indices(A, A.clone(), self.B_l))
        with self.assertRaises(RuntimeError):
            ia.propagate_matmul(A, A.clone(), self.B_l, self.B_u, "rump")

    def test_declines_a_dense_point_matrix(self):
        # The guard that keeps `nonzero` from allocating twice the size of A.
        generator = torch.Generator().manual_seed(30)
        A = torch.rand(2, 4, generator=generator, dtype=torch.float64)
        self._assert_declined(A, A.clone())

    def test_declines_rows_that_balance_out(self):
        # Two non-zeros in one row and none in another: the global count matches
        # the row count, so only the per-row check rejects this.
        A = _one_hot([1, 2], width=4)
        A[0, 3] = 1.0
        A[1] = 0.0
        self._assert_declined(A, A.clone())

    def test_declines_mismatched_dtypes(self):
        A = _one_hot([1, 2], width=4).float()
        self.assertIsNone(ia._point_onehot_row_indices(A, A.clone(), self.B_l))
        with self.assertRaises(RuntimeError):
            ia.propagate_matmul(A, A.clone(), self.B_l, self.B_u, "rump")

    def test_declines_integer_operands(self):
        A = torch.nn.functional.one_hot(torch.tensor([1, 2]), 4)
        self.assertIsNone(ia._point_onehot_row_indices(A, A.clone(), self.B_l))


class TestPointOperandShortCircuit(unittest.TestCase):
    """Rump's algorithm must stay bit-identical when an operand is degenerate."""

    def test_point_a_matches_the_reference(self):
        generator = torch.Generator().manual_seed(6)
        A = torch.randn(5, 4, generator=generator, dtype=torch.float64)
        B_l, B_u = _random_weight_interval(4, 3, seed=7)
        ref_l, ref_u = _reference_rump(A, A.clone(), B_l, B_u)
        H_l, H_u = ia.propagate_matmul_rump(A, A.clone(), B_l, B_u)
        self.assertTrue(torch.equal(H_l, ref_l))
        self.assertTrue(torch.equal(H_u, ref_u))

    def test_point_b_matches_the_reference(self):
        A_l, A_u = _random_weight_interval(5, 4, seed=8)
        generator = torch.Generator().manual_seed(9)
        B = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        ref_l, ref_u = _reference_rump(A_l, A_u, B, B.clone())
        H_l, H_u = ia.propagate_matmul_rump(A_l, A_u, B, B.clone())
        self.assertTrue(torch.equal(H_l, ref_l))
        self.assertTrue(torch.equal(H_u, ref_u))

    def test_general_interval_is_unchanged(self):
        A_l, A_u = _random_weight_interval(5, 4, seed=10)
        B_l, B_u = _random_weight_interval(4, 3, seed=11)
        ref_l, ref_u = _reference_rump(A_l, A_u, B_l, B_u)
        H_l, H_u = ia.propagate_matmul_rump(A_l, A_u, B_l, B_u)
        self.assertTrue(torch.equal(H_l, ref_l))
        self.assertTrue(torch.equal(H_u, ref_u))

    def test_batched_operands_still_take_the_general_path(self):
        A_l, A_u = _random_weight_interval(4, 3, seed=12)
        B_l, B_u = _random_weight_interval(3, 2, seed=13)
        A_l, A_u = A_l.expand(2, 4, 3), A_u.expand(2, 4, 3)
        B_l, B_u = B_l.expand(2, 3, 2), B_u.expand(2, 3, 2)
        ref_l, ref_u = _reference_rump(A_l, A_u, B_l, B_u)
        H_l, H_u = ia.propagate_matmul(A_l, A_u, B_l, B_u, "rump")
        self.assertTrue(torch.equal(H_l, ref_l))
        self.assertTrue(torch.equal(H_u, ref_u))


class TestBoundedModelWithOneHotInputs(unittest.TestCase):
    """The certificate path itself must be unchanged end to end."""

    def _model_bounds(self, x: torch.Tensor, n_states: int):
        torch.manual_seed(14)
        model = torch.nn.Sequential(
            torch.nn.Linear(n_states, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 3),
        ).double()
        bounded_model = IntervalBoundedModel(model)
        for p_l, p_u in zip(bounded_model.param_l, bounded_model.param_u):
            p_l -= 0.01
            p_u += 0.01
        return bounded_model.bound_forward(x, x.clone())

    def test_one_hot_forward_matches_the_dense_computation(self):
        n_states = 16
        x = _one_hot([0, 5, 15, 5], width=n_states)
        fast_l, fast_u = self._model_bounds(x, n_states)
        with mock.patch.object(ia, "_point_onehot_row_indices", return_value=None):
            dense_l, dense_u = self._model_bounds(x, n_states)
        self.assertTrue(torch.allclose(fast_l, dense_l, rtol=0.0, atol=1e-14))
        self.assertTrue(torch.allclose(fast_u, dense_u, rtol=0.0, atol=1e-14))
        # The lookup is exact, so it can only be at least as tight as the
        # midpoint/radius round trip is wide.
        self.assertLessEqual(
            float((fast_u - fast_l).sum()), float((dense_u - dense_l).sum())
        )

    def test_bounds_still_contain_the_nominal_output(self):
        n_states = 16
        x = _one_hot([0, 5, 15, 5], width=n_states)
        torch.manual_seed(14)
        model = torch.nn.Sequential(
            torch.nn.Linear(n_states, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 3),
        ).double()
        bounded_model = IntervalBoundedModel(model)
        for p_l, p_u in zip(bounded_model.param_l, bounded_model.param_u):
            p_l -= 0.01
            p_u += 0.01
        lower, upper = bounded_model.bound_forward(x, x.clone())
        nominal = model(x)
        self.assertTrue(torch.all(lower <= nominal))
        self.assertTrue(torch.all(nominal <= upper))


if __name__ == "__main__":
    unittest.main()
