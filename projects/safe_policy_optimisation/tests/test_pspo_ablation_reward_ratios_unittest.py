"""Regression tests for the globally shifted, seed-paired ranking metric."""

from __future__ import annotations

import math
import unittest

import numpy as np

from projects.safe_policy_optimisation.scripts import (
    compute_pspo_ablation_reward_ratios as ratios,
)


class PairedRewardRatioTests(unittest.TestCase):
    def rewards(self):
        return {
            "pspo": {"negative": [-1.0, 2.0], "zero": [0.0, 1.0]},
            "ablation": {"negative": [-4.0, 1.0], "zero": [1.0, 0.0]},
        }

    def test_global_minimum_not_per_environment(self):
        scores, minimum = ratios.paired_ratios(self.rewards())
        self.assertEqual(minimum, -4.0)
        np.testing.assert_array_equal(scores["pspo"]["zero"], [1.0, 1.0])
        np.testing.assert_allclose(scores["ablation"]["zero"], [5 / 4, 4 / 5])

    def test_denominator_is_matched_seed_not_pspo_mean(self):
        scores, _ = ratios.paired_ratios(self.rewards())
        np.testing.assert_allclose(scores["ablation"]["negative"], [0, 5 / 6])

    def test_pspo_is_exactly_one_with_zero_standard_error(self):
        scores, _ = ratios.paired_ratios(self.rewards())
        summary = ratios.pooled_summary(scores)
        self.assertEqual(summary["pspo"], {"mean": 1.0, "two_se": 0.0, "n": 4})

    def test_shifted_minimum_is_zero_and_ratios_are_not_clipped(self):
        scores, _ = ratios.paired_ratios(self.rewards())
        self.assertEqual(scores["ablation"]["negative"][0], 0)
        self.assertGreater(scores["ablation"]["zero"][0], 1)

    def test_pooled_mean_and_sample_two_standard_errors(self):
        summary = ratios.pooled_summary(
            {"example": {"env1": np.array([0.0, 1.0]), "env2": np.array([1.0, 2.0])}}
        )["example"]
        self.assertEqual(summary["mean"], 1)
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["two_se"], math.sqrt(2 / 3))

    def test_global_additive_offset_leaves_ratios_unchanged(self):
        original = self.rewards()
        shifted = {
            variant: {env: np.asarray(values) + 17 for env, values in per_env.items()}
            for variant, per_env in original.items()
        }
        a, minimum_a = ratios.paired_ratios(original)
        b, minimum_b = ratios.paired_ratios(shifted)
        self.assertEqual(minimum_b, minimum_a + 17)
        for variant in a:
            for env in a[variant]:
                np.testing.assert_array_equal(a[variant][env], b[variant][env])

    def test_zero_reference_is_rejected_without_epsilon_or_dropped_seeds(self):
        with self.assertRaisesRegex(ValueError, "No seeds are dropped"):
            ratios.paired_ratios({"pspo": {"env": [0.0, 1.0]}})

    def test_all_constant_rewards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Undefined PSPO adjusted reward"):
            ratios.paired_ratios(
                {"pspo": {"env": [1.0, 1.0]}, "ablation": {"env": [1.0, 1.0]}}
            )

    def test_missing_environment_is_rejected(self):
        rewards = self.rewards()
        del rewards["ablation"]["zero"]
        with self.assertRaisesRegex(ValueError, "Unmatched environments"):
            ratios.paired_ratios(rewards)

    def test_missing_seed_is_rejected(self):
        rewards = self.rewards()
        rewards["ablation"]["zero"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "Unmatched seed counts"):
            ratios.paired_ratios(rewards)

    def test_nonfinite_rewards_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                rewards = self.rewards()
                rewards["ablation"]["zero"][0] = value
                with self.assertRaisesRegex(ValueError, "finite seed rewards"):
                    ratios.paired_ratios(rewards)

    def test_missing_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "PSPO reference"):
            ratios.paired_ratios({"ablation": {"env": [1, 2]}})

    def test_summary_requires_two_finite_values(self):
        for values in ([1.0], [1.0, math.nan]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "finite observations"):
                    ratios.mean_two_se(values)


if __name__ == "__main__":
    unittest.main()
