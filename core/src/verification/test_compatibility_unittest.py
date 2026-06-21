"""Tests for src.verification.compatibility.check_model_compatibility."""

import unittest

import torch

from src.verification.compatibility import UnsupportedLayerError, check_model_compatibility


class CheckModelCompatibilityTests(unittest.TestCase):
    def test_all_violations_reported_not_just_first(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Sigmoid(),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
        )
        with self.assertRaises(UnsupportedLayerError) as ctx:
            check_model_compatibility(
                model, (torch.nn.Linear, torch.nn.ReLU), method_name="CROWN",
            )
        violations = ctx.exception.violations
        self.assertEqual([idx for idx, _ in violations], [1, 3])
        self.assertIsInstance(violations[0][1], torch.nn.Sigmoid)
        self.assertIsInstance(violations[1][1], torch.nn.MaxPool1d)
        self.assertEqual(ctx.exception.method_name, "CROWN")
        self.assertIn("Sigmoid", str(ctx.exception))
        self.assertIn("MaxPool1d", str(ctx.exception))

    def test_compatible_model_raises_nothing(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        check_model_compatibility(model, (torch.nn.Linear, torch.nn.ReLU), method_name="IBP")


if __name__ == "__main__":
    unittest.main()
