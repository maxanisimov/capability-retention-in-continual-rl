"""Unit tests for multi-environment reward/safety learning curves."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from projects.safe_policy_optimisation.scripts import (
    plot_extended_budget_learning_curves as curves,
)


class LearningCurveAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.baseline_root = root / "baselines"
        self.adaptive_root = root / "pspo"
        self.environment = curves.EnvironmentSpec(
            "test",
            "Test",
            100,
            self.baseline_root,
            self.adaptive_root,
        )
        self.method = curves.METHODS[0]

    def _write_seed(self, seed: int, *, reward: tuple[float, float], safety: tuple[bool, bool]) -> None:
        seed_dir = self.baseline_root / f"seed{seed}"
        evaluation_path = seed_dir / self.method.evaluation_path
        exploration_path = seed_dir / self.method.exploration_path
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        exploration_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "timestep": [0],
                "mean_total_reward": [(reward[0] + reward[1]) / 2],
                "safety_rate": [sum(safety) / 2],
            }
        ).to_csv(evaluation_path, index=False)
        pd.DataFrame(
            {
                "algorithm": [self.method.algorithm_filter] * 2,
                "end_timestep": [20, 40],
                "reward": reward,
                # Deliberately contradict the direct field to test precedence.
                "violated": safety,
                "safe_trajectory": safety,
            }
        ).to_csv(exploration_path, index=False)

    def test_horizon_rounds_up_to_rollout_boundary(self) -> None:
        self.assertEqual(self.environment.horizon, curves.ROLLOUT_SIZE)
        self.assertEqual(curves.ENVIRONMENT_BY_KEY["bridge_crossing_v2"].horizon, 1_601_536)

    def test_evaluation_mean_and_standard_error(self) -> None:
        self._write_seed(0, reward=(1.0, 3.0), safety=(True, False))
        self._write_seed(1, reward=(5.0, 7.0), safety=(True, True))

        aggregate = curves.aggregate_evaluation(
            self.method, environment=self.environment
        ).iloc[0]

        self.assertEqual(aggregate["seed_count"], 2)
        self.assertAlmostEqual(aggregate["reward_mean"], 4.0)
        self.assertAlmostEqual(aggregate["reward_sem"], 2.0)
        self.assertAlmostEqual(aggregate["safety_mean"], 0.75)
        self.assertAlmostEqual(aggregate["safety_sem"], 0.25)

    def test_exploration_prefers_safe_trajectory_over_violated(self) -> None:
        self._write_seed(0, reward=(1.0, 3.0), safety=(True, False))
        self._write_seed(1, reward=(5.0, 7.0), safety=(True, True))

        aggregate = curves.aggregate_exploration(
            self.method,
            environment=self.environment,
            bin_size=curves.ROLLOUT_SIZE,
        ).iloc[0]

        self.assertEqual(aggregate["safety_source"], "safe_trajectory")
        self.assertAlmostEqual(aggregate["reward_mean"], 4.0)
        self.assertAlmostEqual(aggregate["reward_sem"], 2.0)
        self.assertAlmostEqual(aggregate["safety_mean"], 0.75)
        self.assertAlmostEqual(aggregate["safety_sem"], 0.25)


    def test_optional_boolean_parser_preserves_missing_values(self) -> None:
        parsed = curves._parse_optional_bool(
            pd.Series([True, False, "1", "0", "", None])
        )

        self.assertEqual(parsed.iloc[:4].tolist(), [True, False, True, False])
        self.assertTrue(parsed.iloc[4:].isna().all())


class LearningCurveCliTests(unittest.TestCase):
    def test_defaults_to_two_standard_errors(self) -> None:
        args = curves.parse_args([])
        self.assertEqual(args.ci_multiplier, 2.0)
        self.assertIsNone(args.environment)

    def test_single_environment_overrides(self) -> None:
        args = curves.parse_args(
            [
                "--environment",
                "bridge_crossing_v2",
                "--budget",
                "1000000",
                "--pspo-root",
                "/tmp/example-pspo",
            ]
        )
        environment = curves.selected_environments(args)[0]
        self.assertEqual(environment.nominal_budget, 1_000_000)
        self.assertEqual(environment.adaptive_root, Path("/tmp/example-pspo"))


if __name__ == "__main__":
    unittest.main()
