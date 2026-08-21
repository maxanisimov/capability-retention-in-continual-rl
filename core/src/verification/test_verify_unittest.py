"""Tests for the aggregation='none' addition to the multi-label functions in verify.py.

These prove aggregation='none' followed by .min()/.mean() is mathematically identical to
calling aggregation='min'/'mean' directly - the property that makes routing
_get_min_acc and the new verification API through a single shared primitive safe.
"""

import unittest

import torch

from src.verification import verify
from src.IntervalTensor import IntervalTensor


def _sample_logits_and_targets():
    logits_l = torch.tensor([[0.0, 1.0, -1.0], [2.0, 0.5, 0.0], [-1.0, -1.0, 1.0]])
    logits_u = logits_l + 0.5
    targets = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32)
    return IntervalTensor(logits_l, logits_u), targets


class BoundMultiLabelAccuracyAggregationTests(unittest.TestCase):
    def test_none_then_min_matches_min_directly(self):
        logits, targets = _sample_logits_and_targets()
        per_sample = verify.bound_multi_label_accuracy(logits, targets, aggregation="none")
        direct = verify.bound_multi_label_accuracy(logits, targets, aggregation="min")
        self.assertTrue(torch.equal(per_sample.min(), direct))

    def test_none_then_mean_matches_mean_directly(self):
        logits, targets = _sample_logits_and_targets()
        per_sample = verify.bound_multi_label_accuracy(logits, targets, aggregation="none")
        direct = verify.bound_multi_label_accuracy(logits, targets, aggregation="mean")
        self.assertTrue(torch.equal(per_sample.mean(), direct))

    def test_none_returns_per_sample_tensor(self):
        logits, targets = _sample_logits_and_targets()
        per_sample = verify.bound_multi_label_accuracy(logits, targets, aggregation="none")
        self.assertEqual(per_sample.shape, (3,))

    def test_all_mode_requires_every_valid_logit_to_beat_invalid_logits(self):
        logits = IntervalTensor(
            torch.tensor([[2.0, 0.0, 1.0], [3.0, 2.0, 1.0]]),
            torch.tensor([[2.0, 0.0, 1.5], [3.0, 2.0, 1.5]]),
        )
        targets = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)

        any_mode = verify.bound_multi_label_accuracy(
            logits, targets, aggregation="none", mode="any",
        )
        all_mode = verify.bound_multi_label_accuracy(
            logits, targets, aggregation="none", mode="all",
        )

        self.assertTrue(torch.equal(any_mode, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(all_mode, torch.tensor([0.0, 1.0])))

    def test_no_valid_actions_fail_closed(self):
        logits = IntervalTensor(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 2.0]]))
        targets = torch.tensor([[0, 0]], dtype=torch.float32)

        certified = verify.bound_multi_label_accuracy(
            logits, targets, aggregation="none", mode="all",
        )
        margin = verify.bound_multi_label_accuracy_margin(
            logits, targets, aggregation="none", mode="all",
        )

        self.assertTrue(torch.equal(certified, torch.tensor([0.0])))
        self.assertLess(margin.item(), 0.0)


class BoundMultiLabelAccuracyMarginAggregationTests(unittest.TestCase):
    def test_none_then_min_matches_min_directly(self):
        logits, targets = _sample_logits_and_targets()
        per_sample = verify.bound_multi_label_accuracy_margin(logits, targets, aggregation="none")
        direct = verify.bound_multi_label_accuracy_margin(logits, targets, aggregation="min")
        self.assertTrue(torch.equal(per_sample.min(), direct))

    def test_none_then_mean_matches_mean_directly(self):
        logits, targets = _sample_logits_and_targets()
        per_sample = verify.bound_multi_label_accuracy_margin(logits, targets, aggregation="none")
        direct = verify.bound_multi_label_accuracy_margin(logits, targets, aggregation="mean")
        self.assertTrue(torch.equal(per_sample.mean(), direct))

    def test_all_mode_margin_tracks_min_valid_against_max_invalid(self):
        logits = IntervalTensor(
            torch.tensor([[2.0, 0.0, 1.0], [3.0, 2.0, 1.0]]),
            torch.tensor([[2.0, 0.0, 1.5], [3.0, 2.0, 1.5]]),
        )
        targets = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)

        margins = verify.bound_multi_label_accuracy_margin(
            logits, targets, tau=0.1, aggregation="none", mode="all",
        )

        self.assertLess(margins[0].item(), 0.0)
        self.assertGreater(margins[1].item(), 0.0)

    def test_logsumexp_any_matches_manual_cardinality_corrected_formula(self):
        tau = 0.4
        logits_l = torch.tensor([[2.0, 1.0, -1.0], [0.5, -0.5, 1.0]])
        logits_u = logits_l + 0.2
        logits = IntervalTensor(logits_l, logits_u)
        targets = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)

        actual = verify.bound_multi_label_accuracy_margin(
            logits,
            targets,
            tau=tau,
            aggregation="none",
            mode="any",
            surrogate="logsumexp",
        )
        expected = torch.stack(
            [
                tau * torch.logsumexp(logits_l[0, :2] / tau, dim=0)
                - tau * torch.log(torch.tensor(2.0))
                - tau * torch.logsumexp(logits_u[0, 2:] / tau, dim=0),
                logits_l[1, 0]
                - tau * torch.logsumexp(logits_u[1, 1:] / tau, dim=0),
            ]
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_logsumexp_any_has_same_sign_as_probability_surrogate(self):
        logits, targets = _sample_logits_and_targets()
        probability = verify.bound_multi_label_accuracy_margin(
            logits, targets, tau=0.3, aggregation="none", mode="any", surrogate="auto",
        )
        logsumexp = verify.bound_multi_label_accuracy_margin(
            logits,
            targets,
            tau=0.3,
            aggregation="none",
            mode="any",
            surrogate="logsumexp",
        )
        self.assertTrue(torch.equal(probability > 0, logsumexp > 0))

    def test_all_auto_and_logsumexp_are_identical(self):
        logits, targets = _sample_logits_and_targets()
        auto = verify.bound_multi_label_accuracy_margin(
            logits, targets, tau=0.3, aggregation="none", mode="all", surrogate="auto",
        )
        explicit = verify.bound_multi_label_accuracy_margin(
            logits,
            targets,
            tau=0.3,
            aggregation="none",
            mode="all",
            surrogate="logsumexp",
        )
        self.assertTrue(torch.equal(auto, explicit))

    def test_probability_all_checks_every_safe_against_every_unsafe(self):
        logits = IntervalTensor(
            torch.tensor([[3.0, 2.0, 1.0], [3.0, 0.0, 1.0]]),
            torch.tensor([[3.0, 2.0, 1.0], [3.0, 0.0, 1.0]]),
        )
        targets = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)
        margins = verify.bound_multi_label_accuracy_margin(
            logits,
            targets,
            tau=0.5,
            aggregation="none",
            mode="all",
            surrogate="probability",
        )
        self.assertGreater(float(margins[0]), 0.0)
        self.assertLess(float(margins[1]), 0.0)

    def test_probability_all_degenerate_rows_are_finite(self):
        logits = IntervalTensor(
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        )
        targets = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
        margins = verify.bound_multi_label_accuracy_margin(
            logits,
            targets,
            aggregation="none",
            mode="all",
            surrogate="probability",
        )
        self.assertTrue(torch.equal(margins, torch.tensor([1.0, -1.0])))

    def test_logsumexp_avoids_outer_probability_saturation(self):
        targets = torch.tensor([[1, 0, 0]], dtype=torch.float32)

        prob_l = torch.tensor([[20.0, 0.0, -20.0]], requires_grad=True)
        prob_u = prob_l.detach().clone().requires_grad_(True)
        probability = verify.bound_multi_label_accuracy_margin(
            IntervalTensor(prob_l, prob_u),
            targets,
            tau=1.0,
            mode="any",
            surrogate="auto",
        )
        probability.backward()
        probability_grad = prob_l.grad.abs().sum() + prob_u.grad.abs().sum()

        lse_l = prob_l.detach().clone().requires_grad_(True)
        lse_u = prob_u.detach().clone().requires_grad_(True)
        logsumexp = verify.bound_multi_label_accuracy_margin(
            IntervalTensor(lse_l, lse_u),
            targets,
            tau=1.0,
            mode="any",
            surrogate="logsumexp",
        )
        logsumexp.backward()
        lse_grad = lse_l.grad.abs().sum() + lse_u.grad.abs().sum()

        self.assertGreater(lse_grad.item(), probability_grad.item())
        self.assertGreater(lse_grad.item(), 0.5)

    def test_logsumexp_degenerate_rows_have_finite_values_and_gradients(self):
        logits_l = torch.tensor([[1.0, 2.0], [1.0, 2.0]], requires_grad=True)
        logits_u = logits_l.detach().clone().requires_grad_(True)
        targets = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
        margins = verify.bound_multi_label_accuracy_margin(
            IntervalTensor(logits_l, logits_u),
            targets,
            aggregation="none",
            surrogate="logsumexp",
        )
        self.assertTrue(torch.equal(margins, torch.tensor([1.0, -1.0])))
        margins.sum().backward()
        self.assertTrue(torch.isfinite(logits_l.grad).all())
        self.assertTrue(torch.isfinite(logits_u.grad).all())

    def test_invalid_surrogate_and_temperature_raise(self):
        logits, targets = _sample_logits_and_targets()
        with self.assertRaises(ValueError):
            verify.bound_multi_label_accuracy_margin(
                logits, targets, surrogate="unknown",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            verify.bound_multi_label_accuracy_margin(
                logits, targets, tau=0.0, surrogate="logsumexp",
            )


class BoundMultiLabelSoftAccuracyAggregationTests(unittest.TestCase):
    def test_none_then_min_matches_min_directly(self):
        logits, targets = _sample_logits_and_targets()
        per_sample = verify.bound_multi_label_soft_accuracy(logits, targets, aggregation="none")
        direct = verify.bound_multi_label_soft_accuracy(logits, targets, aggregation="min")
        self.assertTrue(torch.equal(per_sample.min(), direct))

    def test_none_then_mean_matches_mean_directly(self):
        logits, targets = _sample_logits_and_targets()
        per_sample = verify.bound_multi_label_soft_accuracy(logits, targets, aggregation="none")
        direct = verify.bound_multi_label_soft_accuracy(logits, targets, aggregation="mean")
        self.assertTrue(torch.equal(per_sample.mean(), direct))


if __name__ == "__main__":
    unittest.main()
