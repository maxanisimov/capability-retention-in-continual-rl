"""Regression tests for _get_min_acc, used to verify its refactor to route through the
new src.verification.api.verify_point preserves exact numeric behavior."""

import unittest

import torch

from abstract_gradient_training.bounded_models import IntervalBoundedModel
from src.interval_utils import _get_min_acc


def _build_model_and_bounds():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
    )
    bounded_model = IntervalBoundedModel(model)
    eps = 0.01
    for p_l, p_u in zip(bounded_model.param_l, bounded_model.param_u):
        p_l.data.sub_(eps)
        p_u.data.add_(eps)
    X = torch.randn(5, 4)
    # multi-hot targets, varying numbers of valid actions per row, including a row with
    # no invalid actions (exercises the `no_invalid` branch in verify.py).
    y = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [1, 1, 1],  # no invalid actions
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return bounded_model, X, y


class GetMinAccRegressionTests(unittest.TestCase):
    """
    These expected values were captured by running this test against the
    pre-refactor implementation of `_get_min_acc` (calling verify.bound_multi_label_accuracy
    directly with aggregation=aggregation), and must remain bit-identical after `_get_min_acc`
    is refactored to route the hard multi-label branch through verify_point.
    """

    def test_hard_multi_label_min_aggregation_matches_pre_refactor_value(self):
        bounded_model, X, y = _build_model_and_bounds()
        acc = _get_min_acc(bounded_model, X, y, multi_label=True, soft=False, aggregation="min")
        self.assertTrue(torch.allclose(acc, torch.tensor(0.0), atol=1e-12))

    def test_hard_multi_label_mean_aggregation_matches_pre_refactor_value(self):
        bounded_model, X, y = _build_model_and_bounds()
        acc = _get_min_acc(bounded_model, X, y, multi_label=True, soft=False, aggregation="mean")
        self.assertTrue(torch.allclose(acc, torch.tensor(0.800000011920929), atol=1e-12))

    def test_soft_multi_label_branches_are_untouched_by_refactor(self):
        bounded_model, X, y = _build_model_and_bounds()
        acc_soft_acc = _get_min_acc(
            bounded_model, X, y, multi_label=True, soft=True,
            multi_label_soft_metric="soft_accuracy", aggregation="mean",
        )
        acc_margin = _get_min_acc(
            bounded_model, X, y, multi_label=True, soft=True,
            multi_label_soft_metric="accuracy_margin", aggregation="mean",
        )
        self.assertTrue(torch.allclose(acc_soft_acc, torch.tensor(0.674409031867981), atol=1e-12))
        self.assertTrue(torch.allclose(acc_margin, torch.tensor(0.05774230882525444), atol=1e-12))

    def test_single_label_branch_is_untouched_by_refactor(self):
        bounded_model, X, _ = _build_model_and_bounds()
        targets = torch.tensor([0, 1, 2, 0, 1])
        acc = _get_min_acc(bounded_model, X, targets, multi_label=False, soft=False)
        self.assertTrue(torch.allclose(acc, torch.tensor(0.4000000059604645), atol=1e-12))


if __name__ == "__main__":
    unittest.main()
