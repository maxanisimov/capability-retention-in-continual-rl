"""Tests for the Rashomon-set computation API: the order-statistic aggregation,
temperature calibration, `_get_min_acc`, `_certify_groups`, and `compute_rashomon_set`."""

import unittest

import torch

from abstract_gradient_training.bounded_models import IntervalBoundedModel
from src.IntervalTensor import IntervalTensor
import src.verification.verify as verify
from src.interval_utils import (
    _calibrate_temperature,
    _certify_groups,
    _get_min_acc,
    _order_statistic_k,
    _order_statistic_select,
    compute_rashomon_set,
)
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


def _raw_logits(bounded_model, X_l, X_u):
    return IntervalTensor(*bounded_model.bound_forward(X_l, X_u))


def _train_to_fit(model, x, y, steps=200, lr=0.05):
    """Briefly train `model` so its nominal weights actually satisfy a reasonable accuracy
    target - calibration (correctly) refuses to find a temperature for a model that
    doesn't already meet its target accuracy at any temperature, so smoke tests of the
    end-to-end pipeline need a model that's actually fit to its data, not a random init."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model


class OrderStatisticTests(unittest.TestCase):
    """`_order_statistic_k`/`_order_statistic_select`: the exact-count generalization of
    'min' (target_accuracy=1.0) to 'tolerate some fraction of failures'."""

    def test_target_accuracy_one_gives_literal_min_rank(self):
        self.assertEqual(_order_statistic_k(1.0, n=10), 1)

    def test_target_accuracy_zero_gives_max_rank(self):
        self.assertEqual(_order_statistic_k(0.0, n=10), 10)

    def test_fractional_target_accuracy(self):
        # 90% of 10 samples must pass -> at most 1 may fail -> rank 2 (the 2nd smallest)
        self.assertEqual(_order_statistic_k(0.9, n=10), 2)
        # 50% of 10 -> at most 5 may fail -> rank 6
        self.assertEqual(_order_statistic_k(0.5, n=10), 6)

    def test_n_one_always_clamps_to_rank_one(self):
        for p in (0.0, 0.3, 0.7, 1.0):
            self.assertEqual(_order_statistic_k(p, n=1), 1)

    def test_select_matches_kthvalue_directly(self):
        values = torch.tensor([5.0, 1.0, 3.0, 4.0, 2.0])
        selected = _order_statistic_select(values, target_accuracy=0.9)  # n=5, k=1 -> min
        self.assertTrue(torch.allclose(selected, torch.tensor(1.0)))
        selected_k3 = _order_statistic_select(values, target_accuracy=0.5)  # k = int(0.5*5)+1 = 3
        self.assertTrue(torch.allclose(selected_k3, torch.tensor(3.0)))

    def test_select_is_differentiable_with_one_hot_gradient(self):
        values = torch.tensor([5.0, 1.0, 3.0, 4.0, 2.0], requires_grad=True)
        selected = _order_statistic_select(values, target_accuracy=1.0)  # picks index 1 (value 1.0)
        selected.backward()
        expected_grad = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(values.grad, expected_grad))


class GetMinAccTests(unittest.TestCase):
    """`_get_min_acc` should equal the order statistic (implied by `target_accuracy`) of
    the same per-sample arrays produced directly by `verify.py`'s `aggregation='none'`."""

    def test_hard_multi_label_matches_independently_computed_order_statistic(self):
        bounded_model, X, y = _build_model_and_bounds()
        per_sample = verify.bound_multi_label_accuracy(
            _raw_logits(bounded_model, X, X), y, lower=True, aggregation="none",
        )
        for target_accuracy in (1.0, 0.8, 0.6, 0.4, 0.2):
            accuracy = AccuracyRequirement(target_accuracy=target_accuracy)
            acc = _get_min_acc(
                bounded_model, X, X, y, accuracy, group=None, temperature=10.0,
                multi_label=True, soft=False,
            )
            expected = _order_statistic_select(per_sample, target_accuracy)
            self.assertTrue(torch.allclose(acc, expected), msg=f"target_accuracy={target_accuracy}")

    def test_soft_multi_label_matches_independently_computed_order_statistic(self):
        bounded_model, X, y = _build_model_and_bounds()
        T = 7.0
        per_sample = verify.bound_multi_label_accuracy_margin(
            _raw_logits(bounded_model, X, X), y, T=T, lower=True, aggregation="none",
        )
        for target_accuracy in (1.0, 0.8, 0.6, 0.4, 0.2):
            accuracy = AccuracyRequirement(target_accuracy=target_accuracy)
            acc = _get_min_acc(
                bounded_model, X, X, y, accuracy, group=None, temperature=T,
                multi_label=True, soft=True,
            )
            expected = _order_statistic_select(per_sample, target_accuracy)
            self.assertTrue(torch.allclose(acc, expected), msg=f"target_accuracy={target_accuracy}")

    def test_target_accuracy_one_matches_old_strict_min_behavior(self):
        # target_accuracy=1.0 -> k=1 -> the literal minimum, i.e. the old aggregation="min".
        bounded_model, X, y = _build_model_and_bounds()
        per_sample = verify.bound_multi_label_accuracy(
            _raw_logits(bounded_model, X, X), y, lower=True, aggregation="none",
        )
        acc = _get_min_acc(
            bounded_model, X, X, y, AccuracyRequirement(target_accuracy=1.0),
            group=None, temperature=10.0, multi_label=True, soft=False,
        )
        self.assertTrue(torch.allclose(acc, per_sample.min()))

    def test_single_label_is_one_hot_special_case_of_multi_label(self):
        bounded_model, X, _ = _build_model_and_bounds()
        targets = torch.tensor([0, 1, 2, 0, 1])
        one_hot = torch.nn.functional.one_hot(targets, num_classes=3).float()
        per_sample = verify.bound_multi_label_accuracy(
            _raw_logits(bounded_model, X, X), one_hot, lower=True, aggregation="none",
        )
        accuracy = AccuracyRequirement(target_accuracy=0.6)
        acc = _get_min_acc(
            bounded_model, X, X, targets, accuracy, group=None, temperature=10.0,
            multi_label=False, soft=False,
        )
        expected = _order_statistic_select(per_sample, 0.6)
        self.assertTrue(torch.allclose(acc, expected))


class GetMinAccIntervalInputTests(unittest.TestCase):
    """Tests for the X_l != X_u path (input-region certification, not just points)."""

    def test_widening_input_interval_can_only_loosen_hard_accuracy(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(target_accuracy=0.0)
        point_acc = _get_min_acc(
            bounded_model, X, X, y, accuracy, group=None, temperature=10.0,
            multi_label=True, soft=False,
        )

        eps = 0.05
        X_l, X_u = X - eps, X + eps
        interval_acc = _get_min_acc(
            bounded_model, X_l, X_u, y, accuracy, group=None, temperature=10.0,
            multi_label=True, soft=False,
        )

        # widening the input region can only ever certify fewer (or equal) samples
        self.assertLessEqual(interval_acc.item(), point_acc.item() + 1e-12)

    def test_zero_width_interval_matches_point_input(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(target_accuracy=0.0)
        point_acc = _get_min_acc(
            bounded_model, X, X, y, accuracy, group=None, temperature=10.0,
            multi_label=True, soft=False,
        )
        interval_acc = _get_min_acc(
            bounded_model, X, X.clone(), y, accuracy, group=None, temperature=10.0,
            multi_label=True, soft=False,
        )
        self.assertTrue(torch.allclose(point_acc, interval_acc, atol=1e-12))


class CertifyGroupsTests(unittest.TestCase):
    """Tests for `_certify_groups`, which underlies per-group certificate reporting."""

    def test_no_group_by_produces_single_global_group_certificate(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(target_accuracy=0.0)
        certs = _certify_groups(
            bounded_model, X, X, y, accuracy, groups=[None], group_by=None,
            multi_label=True, context_mask=None, temperatures={None: 10.0},
        )
        self.assertEqual(len(certs), 1)
        self.assertIsNone(certs[0].group)

    def test_group_by_splits_certificates_per_group(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(target_accuracy=0.0)
        # group rows by whether the first valid-action bit is set
        group_by = lambda y: y[:, 0].long()
        groups = sorted(group_by(y).unique().tolist())
        certs = _certify_groups(
            bounded_model, X, X, y, accuracy, groups=groups, group_by=group_by,
            multi_label=True, context_mask=None,
            temperatures={g: 10.0 for g in groups},
        )
        self.assertEqual({c.group for c in certs}, set(groups))


class AccuracyRequirementResolveTests(unittest.TestCase):
    def test_shared_float_target(self):
        accuracy = AccuracyRequirement(target_accuracy=0.8)
        self.assertEqual(accuracy.resolve(0), 0.8)
        self.assertEqual(accuracy.resolve(1), 0.8)

    def test_per_group_dict_target(self):
        accuracy = AccuracyRequirement(target_accuracy={0: 0.5, 1: 0.9})
        self.assertEqual(accuracy.resolve(0), 0.5)
        self.assertEqual(accuracy.resolve(1), 0.9)


class CalibrateTemperatureTests(unittest.TestCase):
    def test_calibrates_a_temperature_from_the_doubling_ladder(self):
        bounded_model, X, y = _build_model_and_bounds()
        accuracy = AccuracyRequirement(target_accuracy=0.6)
        temperatures = _calibrate_temperature(
            bounded_model, X, X, y, accuracy, groups=[None], group_by=None,
            multi_label=True, context_mask=None,
        )
        self.assertEqual(set(temperatures), {None})
        candidates = []
        T = 0.1
        while T < 100.0:
            candidates.append(T)
            T *= 2.0
        candidates.append(100.0)
        self.assertIn(temperatures[None], candidates)

    def test_raises_when_no_temperature_in_range_works(self):
        bounded_model, X, y = _build_model_and_bounds()
        # target_accuracy=1.0 -> k=1 -> every sample must pass. Sample index 1 in this
        # fixture has a failing hard condition (its worst-case valid logit never exceeds
        # the worst-case invalid logit), so its margin is negative at every T - and gets
        # *more* negative as T grows, since the softmax concentrates onto the larger
        # (invalid) worst-case logit. No T can rescue it: this is independent of search range.
        accuracy = AccuracyRequirement(target_accuracy=1.0)
        with self.assertRaises(ValueError) as ctx:
            _calibrate_temperature(
                bounded_model, X, X, y, accuracy, groups=[None], group_by=None,
                multi_label=True, context_mask=None,
            )
        message = str(ctx.exception)
        self.assertIn("1.0", message)
        self.assertIn("0.1", message)
        self.assertIn("100", message)

    def test_groups_with_different_targets_calibrate_independently(self):
        bounded_model, X, y = _build_model_and_bounds()
        group_by = lambda y: y[:, 0].long()
        groups = sorted(group_by(y).unique().tolist())
        # group 0 (y[:, 0]==0, indices {1, 4}) contains a sample whose hard condition fails
        # at every T - give it a loose target (only the better of its 2 samples need pass).
        # group 1 (y[:, 0]==1, indices {0, 2, 3}) is all hard-certifiable - give it a tight target.
        accuracy = AccuracyRequirement(target_accuracy={groups[0]: 0.3, groups[1]: 0.99})
        temperatures = _calibrate_temperature(
            bounded_model, X, X, y, accuracy, groups, group_by,
            multi_label=True, context_mask=None,
        )
        self.assertEqual(set(temperatures), set(groups))


class ComputeRashomonSetSmokeTests(unittest.TestCase):
    """End-to-end smoke tests for compute_rashomon_set: interval inputs, multi-group
    accuracy requirements, internal calibration, and an alternate certification method."""

    def test_interval_dataset_with_grouping_and_alternate_certification_method(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))

        n = 12
        x = torch.randn(n, 3)
        y = (x[:, 0] + x[:, 1] - x[:, 2] > 0).long()
        _train_to_fit(model, x, y)
        eps = 0.01
        x_l, x_u = x - eps, x + eps
        dataset = torch.utils.data.TensorDataset(x_l, x_u, y)

        accuracy = AccuracyRequirement(target_accuracy={0: 0.4, 1: 0.4})

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
        self.assertEqual(set(result.temperatures), {0, 1})
        for cert in result.certificates[0]:
            self.assertIsInstance(cert.min_soft_acc, float)
            self.assertIsInstance(cert.min_hard_acc, float)

    def test_checkpointing_produces_one_certificate_list_per_checkpoint(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        n = 8
        x = torch.randn(n, 3)
        y = torch.zeros(n, dtype=torch.long)
        _train_to_fit(model, x, y)
        dataset = torch.utils.data.TensorDataset(x, y)
        accuracy = AccuracyRequirement(target_accuracy=0.5)

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

    def test_explicit_temperatures_override_skips_calibration(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        n = 8
        x = torch.randn(n, 3)
        y = torch.zeros(n, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x, y)
        accuracy = AccuracyRequirement(target_accuracy=0.5)

        # an arbitrary value the auto-search would not necessarily land on, to prove it's
        # used verbatim rather than recalibrated.
        forced_temperature = 0.1234
        result = compute_rashomon_set(
            model, dataset, accuracy,
            batch_size=n, certificate_samples=n, n_iters=2,
            temperatures={None: forced_temperature},
        )
        self.assertEqual(result.temperatures, {None: forced_temperature})

    def test_mismatched_temperatures_override_raises(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        n = 8
        x = torch.randn(n, 3)
        y = torch.zeros(n, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(x, y)
        accuracy = AccuracyRequirement(target_accuracy=0.5)

        with self.assertRaises(ValueError):
            compute_rashomon_set(
                model, dataset, accuracy,
                batch_size=n, certificate_samples=n, n_iters=2,
                temperatures={0: 1.0},  # wrong key: groups is [None] here
            )


if __name__ == "__main__":
    unittest.main()
