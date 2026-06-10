"""Smoke tests for legacy experiment pipeline compatibility delegates."""

from __future__ import annotations

import importlib
from pathlib import Path
import unittest


def _module_file(module: object) -> Path:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise AssertionError(f"{module!r} does not expose __file__.")
    return Path(module_file).resolve()


class CompatibilityWrapperTest(unittest.TestCase):
    def test_legacy_package_imports_resolve_to_canonical_packages(self) -> None:
        module_pairs = [
            (
                "experiments.pipelines.frozenlake",
                "experiments.pipelines.behaviour_retention.frozenlake",
            ),
            (
                "experiments.pipelines.lunarlander",
                "experiments.pipelines.behaviour_retention.lunarlander",
            ),
            (
                "experiments.pipelines.frozenlake_safety",
                "experiments.pipelines.behaviour_retention.frozenlake_safety_constrained",
            ),
            (
                "experiments.pipelines.frozenlake_shield_safety",
                "experiments.pipelines.safety.frozenlake",
            ),
            (
                "experiments.pipelines.frozenlake_slippery_shield_safety",
                "experiments.pipelines.safety.frozenlake_slippery",
            ),
            (
                "experiments.pipelines.lavacrossing_shield_safety",
                "experiments.pipelines.safety.lavacrossing",
            ),
        ]

        for legacy_name, canonical_name in module_pairs:
            with self.subTest(legacy_name=legacy_name):
                legacy = importlib.import_module(legacy_name)
                canonical = importlib.import_module(canonical_name)
                self.assertEqual(_module_file(legacy), _module_file(canonical))

    def test_legacy_module_imports_resolve_to_canonical_modules(self) -> None:
        module_pairs = [
            (
                "experiments.pipelines.frozenlake.run_experiment",
                "experiments.pipelines.behaviour_retention.frozenlake.run_experiment",
            ),
            (
                "experiments.pipelines.lunarlander.core.orchestration.run_paths",
                "experiments.pipelines.behaviour_retention.lunarlander.core.orchestration.run_paths",
            ),
            (
                "experiments.pipelines.frozenlake_safety.core.safety",
                "experiments.pipelines.behaviour_retention.frozenlake_safety_constrained.core.safety",
            ),
            (
                "experiments.pipelines.frozenlake_shield_safety.core.pipeline",
                "experiments.pipelines.safety.frozenlake.core.pipeline",
            ),
            (
                "experiments.pipelines.frozenlake_slippery_shield_safety.core.pipeline",
                "experiments.pipelines.safety.frozenlake_slippery.core.pipeline",
            ),
            (
                "experiments.pipelines.lavacrossing_shield_safety.run_experiment",
                "experiments.pipelines.safety.lavacrossing.run_experiment",
            ),
        ]

        for legacy_name, canonical_name in module_pairs:
            with self.subTest(legacy_name=legacy_name):
                legacy = importlib.import_module(legacy_name)
                canonical = importlib.import_module(canonical_name)
                self.assertEqual(_module_file(legacy), _module_file(canonical))


if __name__ == "__main__":
    unittest.main()
