from __future__ import annotations

import unittest

import torch
from torch.utils.data import TensorDataset

from projects.safe_crl.utils.distillation_ppo import demonstration_metrics


class DistillationPPOTests(unittest.TestCase):
    def test_demonstration_metrics_accept_one_hot_labels(self) -> None:
        actor = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            actor[0].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        dataset = TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

        metrics = demonstration_metrics(actor, dataset)

        self.assertEqual(metrics["source_demo_accuracy"], 1.0)
        self.assertGreater(metrics["source_demo_mean_action_probability"], 0.8)
        self.assertLess(metrics["source_demo_cross_entropy"], 0.2)

    def test_demonstration_metrics_accept_class_labels(self) -> None:
        actor = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            actor[0].weight.copy_(torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
        dataset = TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([1, 0]),
        )

        metrics = demonstration_metrics(actor, dataset)

        self.assertEqual(metrics["source_demo_accuracy"], 1.0)

    def test_demonstration_metrics_reject_empty_dataset(self) -> None:
        actor = torch.nn.Sequential(torch.nn.Linear(2, 2))
        dataset = TensorDataset(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long))

        with self.assertRaises(ValueError):
            demonstration_metrics(actor, dataset)


if __name__ == "__main__":
    unittest.main()
