"""Regression and behavioral tests for the Rashomon-set computation API:
`_get_min_acc`, `_certify_groups`, and `compute_rashomon_set`."""

import unittest

import torch

from abstract_gradient_training.bounded_models import IntervalBoundedModel
from src.interval_utils import _get_min_acc, _certify_groups, compute_rashomon_set
from src.rashomon_spec import AccuracyRequirement


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
    is refactored to route the hard multi-label branch through verify_point, and again after
    the accuracy parameters were consolidated into `AccuracyRequirement`.
    """

    def test_hard_multi_label_min_aggregation_matches_pre_refactor_value(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(soft_min=0.0, aggregation="min")
        acc = _get_min_acc(bounded_model, X, X, y, accuracy, multi_label=True, soft=False)
        self.assertTrue(torch.allclose(acc, torch.tensor(0.0), atol=1e-12))

    def test_hard_multi_label_mean_aggregation_matches_pre_refactor_value(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(soft_min=0.0, aggregation="mean")
        acc = _get_min_acc(bounded_model, X, X, y, accuracy, multi_label=True, soft=False)
        self.assertTrue(torch.allclose(acc, torch.tensor(0.800000011920929), atol=1e-12))

    def test_soft_multi_label_branches_are_untouched_by_refactor(self):
        bounded_model, X, y = _build_model_and_bounds()
        acc_soft_acc = _get_min_acc(
            bounded_model, X, X, y,
            AccuracyRequirement(soft_min=0.0, soft_metric="soft_accuracy", aggregation="mean"),
            multi_label=True, soft=True,
        )
        acc_margin = _get_min_acc(
            bounded_model, X, X, y,
            AccuracyRequirement(soft_min=0.0, soft_metric="accuracy_margin", aggregation="mean"),
            multi_label=True, soft=True,
        )
        self.assertTrue(torch.allclose(acc_soft_acc, torch.tensor(0.674409031867981), atol=1e-12))
        self.assertTrue(torch.allclose(acc_margin, torch.tensor(0.05774230882525444), atol=1e-12))

    def test_single_label_branch_is_untouched_by_refactor(self):
        bounded_model, X, _ = _build_model_and_bounds()
        targets = torch.tensor([0, 1, 2, 0, 1])
        accuracy = AccuracyRequirement(soft_min=0.0)
        acc = _get_min_acc(bounded_model, X, X, targets, accuracy, multi_label=False, soft=False)
        self.assertTrue(torch.allclose(acc, torch.tensor(0.4000000059604645), atol=1e-12))


class GetMinAccIntervalInputTests(unittest.TestCase):
    """Tests for the X_l != X_u path (input-region certification, not just points)."""

    def test_widening_input_interval_can_only_loosen_hard_accuracy(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(soft_min=0.0, aggregation="mean")
        point_acc = _get_min_acc(bounded_model, X, X, y, accuracy, multi_label=True, soft=False)

        eps = 0.05
        X_l, X_u = X - eps, X + eps
        interval_acc = _get_min_acc(bounded_model, X_l, X_u, y, accuracy, multi_label=True, soft=False)

        # widening the input region can only ever certify fewer (or equal) samples
        self.assertLessEqual(interval_acc.item(), point_acc.item() + 1e-12)

    def test_zero_width_interval_matches_point_input(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(soft_min=0.0, aggregation="mean")
        point_acc = _get_min_acc(bounded_model, X, X, y, accuracy, multi_label=True, soft=False)
        interval_acc = _get_min_acc(bounded_model, X, X.clone(), y, accuracy, multi_label=True, soft=False)
        self.assertTrue(torch.allclose(point_acc, interval_acc, atol=1e-12))


class CertifyGroupsTests(unittest.TestCase):
    """Tests for `_certify_groups`, which underlies per-group certificate reporting."""

    def test_no_group_by_produces_single_global_group_certificate(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(soft_min=0.0, aggregation="mean")
        certs = _certify_groups(
            bounded_model, X, X, y, accuracy, groups=[None], group_by=None,
            multi_label=True, context_mask=None,
        )
        self.assertEqual(len(certs), 1)
        self.assertIsNone(certs[0].group)

    def test_group_by_splits_certificates_per_group(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(soft_min=0.0, aggregation="mean")
        # group rows by whether the first valid-action bit is set
        group_by = lambda y: y[:, 0].long()
        groups = sorted(group_by(y).unique().tolist())
        certs = _certify_groups(
            bounded_model, X, X, y, accuracy, groups=groups, group_by=group_by,
            multi_label=True, context_mask=None,
        )
        self.assertEqual({c.group for c in certs}, set(groups))


class AccuracyRequirementResolveTests(unittest.TestCase):
    def test_shared_float_limits(self):
        accuracy = AccuracyRequirement(soft_min=0.8, hard_min=0.95)
        self.assertEqual(accuracy.resolve(0), (0.8, 0.95))
        self.assertEqual(accuracy.resolve(1), (0.8, 0.95))

    def test_hard_min_defaults_to_soft_min(self):
        accuracy = AccuracyRequirement(soft_min=0.7)
        self.assertEqual(accuracy.resolve(0), (0.7, 0.7))

    def test_per_group_dict_limits(self):
        accuracy = AccuracyRequirement(soft_min={0: 0.5, 1: 0.9}, hard_min={0: 0.6, 1: 1.0})
        self.assertEqual(accuracy.resolve(0), (0.5, 0.6))
        self.assertEqual(accuracy.resolve(1), (0.9, 1.0))


class ComputeRashomonSetSmokeTests(unittest.TestCase):
    """End-to-end smoke tests for compute_rashomon_set: interval inputs, multi-group
    accuracy requirements, and an alternate certification method, all together."""

    def test_interval_dataset_with_grouping_and_alternate_certification_method(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))

        n = 12
        x = torch.randn(n, 3)
        eps = 0.01
        x_l, x_u = x - eps, x + eps
        y = torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n - n // 2, dtype=torch.long)])
        dataset = torch.utils.data.TensorDataset(x_l, x_u, y)

        accuracy = AccuracyRequirement(soft_min={0: 0.4, 1: 0.4}, hard_min={0: 0.4, 1: 0.4})

        result = compute_rashomon_set(
            model, dataset, accuracy,
            batch_size=n, certificate_samples=n, n_iters=3,
            has_input_intervals=True, group_by=lambda y: y,
            certification_method="CROWN",
        )

        self.assertEqual(len(result.bounded_models), 1)
        self.assertEqual(len(result.certificates), 1)
        cert_groups = {c.group for c in result.certificates[0]}
        self.assertEqual(cert_groups, {0, 1})
        for cert in result.certificates[0]:
            self.assertIsInstance(cert.min_soft_acc, float)
            self.assertIsInstance(cert.min_hard_acc, float)

    def test_checkpointing_produces_one_certificate_list_per_checkpoint(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        n = 8
        x = torch.randn(n, 3)
        y = torch.zeros(n, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x, y)
        accuracy = AccuracyRequirement(soft_min=0.5, hard_min=0.5)

        result = compute_rashomon_set(
            model, dataset, accuracy,
            batch_size=n, certificate_samples=n, n_iters=4, checkpoint=2,
        )
        # n_iters=4, checkpoint=2 -> checkpoints recorded at iter 2 plus the final model
        self.assertEqual(len(result.bounded_models), 2)
        self.assertEqual(len(result.certificates), 2)
        for certs in result.certificates:
            self.assertEqual(len(certs), 1)
            self.assertIsNone(certs[0].group)


if __name__ == "__main__":
    unittest.main()
