"""Tests for src.verification.registry."""

import unittest

import torch

from abstract_gradient_training.bounded_models import CROWNBoundedModel, IntervalBoundedModel
from src.verification.api import build_bounded_model
from src.verification.registry import (
    VerificationMethod,
    available_methods,
    get_method,
    register_method,
)


class RegistryTests(unittest.TestCase):
    def test_get_method_known_names(self):
        ibp = get_method("IBP")
        self.assertIs(ibp.bounded_model_cls, IntervalBoundedModel)

        crown = get_method("CROWN")
        self.assertIs(crown.bounded_model_cls, CROWNBoundedModel)
        self.assertEqual(crown.default_kwargs["relu_relaxation"], "zero")
        self.assertEqual(crown.default_kwargs["tanh_relaxation"], "fixed")

        alpha_crown = get_method("alpha-CROWN")
        self.assertIs(alpha_crown.bounded_model_cls, CROWNBoundedModel)
        self.assertEqual(alpha_crown.default_kwargs["relu_relaxation"], "optimizable")
        self.assertEqual(alpha_crown.default_kwargs["tanh_relaxation"], "optimizable")

    def test_get_method_unknown_raises_with_available_methods_listed(self):
        with self.assertRaises(ValueError) as ctx:
            get_method("bogus-method")
        self.assertIn("IBP", str(ctx.exception))

    def test_register_method_custom(self):
        name = "test-only-method"
        method = VerificationMethod(
            name=name,
            bounded_model_cls=IntervalBoundedModel,
            supported_modules=(torch.nn.Linear,),
        )
        register_method(method)
        self.assertIn(name, available_methods())
        self.assertIs(get_method(name), method)

        with self.assertRaises(ValueError):
            register_method(method, overwrite=False)
        # should not raise
        register_method(method, overwrite=True)

    def test_register_method_custom_usable_via_build_bounded_model(self):
        name = "test-only-method-usable"
        register_method(
            VerificationMethod(
                name=name,
                bounded_model_cls=IntervalBoundedModel,
                supported_modules=(torch.nn.Linear,),
            )
        )
        model = torch.nn.Sequential(torch.nn.Linear(3, 4))
        bounded_model = build_bounded_model(model, name)
        self.assertIsInstance(bounded_model, IntervalBoundedModel)


if __name__ == "__main__":
    unittest.main()
