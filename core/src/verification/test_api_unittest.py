"""Tests for src.verification.api (build_bounded_model, verify_point, verify_dataset)."""

import unittest

import torch

from abstract_gradient_training.bounded_models import IntervalBoundedModel
from src.verification import verify
from src.verification.api import AdmissibleSet, build_bounded_model, verify_dataset, verify_point
from src.verification.compatibility import UnsupportedLayerError
from src.verification.interval_tensor import IntervalTensor


def _build_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 4))


class AdmissibleSetTests(unittest.TestCase):
    def test_valid_indices_broadcasts_to_batch(self):
        admissible = AdmissibleSet(n_classes=4, valid_indices=[0, 2])
        mask = admissible.as_multi_hot(batch_size=3, device=torch.device("cpu"))
        expected = torch.tensor([[True, False, True, False]] * 3)
        self.assertTrue(torch.equal(mask, expected))

    def test_multi_hot_per_row(self):
        multi_hot = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0]])
        admissible = AdmissibleSet(n_classes=4, multi_hot=multi_hot)
        mask = admissible.as_multi_hot(batch_size=2, device=torch.device("cpu"))
        self.assertTrue(torch.equal(mask, multi_hot.bool()))


class BuildBoundedModelTests(unittest.TestCase):
    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            build_bounded_model(_build_model(), "bogus")

    def test_incompatible_layer_raises_full_report(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Sigmoid())
        with self.assertRaises(UnsupportedLayerError) as ctx:
            build_bounded_model(model, "CROWN")
        self.assertEqual(len(ctx.exception.violations), 1)

    def test_point_network_is_not_trainable_with_no_param_interval(self):
        model = _build_model()
        bounded_model = build_bounded_model(model, "IBP")
        self.assertFalse(bounded_model.trainable)

    def test_param_interval_is_applied(self):
        model = _build_model()
        nominal = [p.detach().clone() for p in model.parameters()]
        param_l = [p - 0.01 for p in nominal]
        param_u = [p + 0.01 for p in nominal]
        bounded_model = build_bounded_model(model, "IBP", param_l=param_l, param_u=param_u)
        for p_l, p_u, p_n in zip(bounded_model.param_l, bounded_model.param_u, nominal):
            self.assertTrue(torch.allclose(p_l, p_n - 0.01))
            self.assertTrue(torch.allclose(p_u, p_n + 0.01))


class VerifyPointTests(unittest.TestCase):
    def test_point_only_matches_direct_ibp_and_verify_call(self):
        model = _build_model()
        bounded_model = build_bounded_model(model, "IBP")
        x = torch.randn(4, 3)
        admissible = AdmissibleSet(n_classes=4, valid_indices=[0, 1])

        result = verify_point(bounded_model, x, admissible)

        logits = IntervalTensor(*bounded_model.bound_forward(x, x))
        mask = torch.zeros(4, 4)
        mask[:, [0, 1]] = 1.0
        expected = verify.bound_multi_label_accuracy(logits, mask, lower=True, aggregation="none").bool()
        self.assertTrue(torch.equal(result.certified, expected))

    def test_input_and_param_interval_combine_in_single_bound_forward_call(self):
        model = _build_model()
        nominal = [p.detach().clone() for p in model.parameters()]
        param_l = [p - 0.01 for p in nominal]
        param_u = [p + 0.01 for p in nominal]
        bounded_model = build_bounded_model(model, "IBP", param_l=param_l, param_u=param_u)

        x = torch.randn(4, 3)
        eps = 0.02
        admissible = AdmissibleSet(n_classes=4, valid_indices=[0, 1])

        result = verify_point(bounded_model, x, admissible, x_l=x - eps, x_u=x + eps)

        # cross-check against manually calling bound_forward with the same bounds
        manual_l, manual_u = bounded_model.bound_forward(x - eps, x + eps)
        self.assertTrue(torch.equal(result.logits_l, manual_l))
        self.assertTrue(torch.equal(result.logits_u, manual_u))


class VerifyDatasetTests(unittest.TestCase):
    def test_chunked_matches_unchunked(self):
        model = _build_model()
        bounded_model = build_bounded_model(model, "IBP")
        X = torch.randn(7, 3)
        mask = torch.zeros(7, 4)
        mask[:, 0] = 1.0

        unchunked = verify_dataset(bounded_model, X, mask)
        chunked = verify_dataset(bounded_model, X, mask, batch_size=3)

        self.assertTrue(torch.equal(unchunked.certified, chunked.certified))
        self.assertTrue(torch.equal(unchunked.logits_l, chunked.logits_l))
        self.assertTrue(torch.equal(unchunked.logits_u, chunked.logits_u))

    def test_matches_looped_verify_point(self):
        model = _build_model()
        bounded_model = build_bounded_model(model, "IBP")
        X = torch.randn(5, 3)
        mask = torch.zeros(5, 4)
        mask[:, 1] = 1.0

        dataset_result = verify_dataset(bounded_model, X, mask)

        per_sample_certified = []
        for i in range(X.shape[0]):
            res = verify_point(
                bounded_model, X[i : i + 1],
                AdmissibleSet(n_classes=4, multi_hot=mask[i : i + 1]),
            )
            per_sample_certified.append(res.certified)
        looped = torch.cat(per_sample_certified)

        self.assertTrue(torch.equal(dataset_result.certified, looped))


if __name__ == "__main__":
    unittest.main()
