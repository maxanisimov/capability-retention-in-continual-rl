"""Tests for the aggregation='none' addition to the multi-label functions in verify.py.

These prove aggregation='none' followed by .min()/.mean() is mathematically identical to
calling aggregation='min'/'mean' directly - the property that makes routing
_get_min_acc and the new verification API through a single shared primitive safe.
"""

import unittest

import torch

from src.verification import verify
from src.verification.interval_tensor import IntervalTensor


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
