"""Tests for the discrete MountainCar continuous-state shield."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from continuous_state_shields import MountainCarShield, ShieldDatasetCollector


class MountainCarShieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shield = MountainCarShield()

    def test_central_state_allows_every_action(self) -> None:
        self.assertEqual(self.shield.get_safe_actions([-0.5, 0.0]), [0, 1, 2])

    def test_left_boundary_contact_is_reported_and_push_right_is_forced(self) -> None:
        safe, info = self.shield.get_safe_actions(
            [np.float32(-1.2), -0.01], return_info=True
        )
        self.assertEqual(safe, [2])
        self.assertTrue(info["shield_active"])
        self.assertEqual(info["region"], "unsafe_boundary_safety_critical")
        self.assertIn("left_boundary_contact", info["violated_constraints"])
        self.assertEqual(self.shield.shield_action([-1.2, -0.01], 0), 2)

    def test_safety_critical_region_only_allows_push_right(self) -> None:
        for position in (-1.2, -1.1, -1.0):
            with self.subTest(position=position):
                safe, info = self.shield.get_safe_actions(
                    [position, -0.01], return_info=True
                )
                self.assertEqual(safe, [2])
                expected_region = (
                    "unsafe_boundary_safety_critical"
                    if position == -1.2
                    else "safety_critical"
                )
                self.assertEqual(info["region"], expected_region)
                expected_violations = (
                    ["left_boundary_contact"] if position == -1.2 else []
                )
                self.assertEqual(info["violated_constraints"], expected_violations)

    def test_safe_region_allows_every_action(self) -> None:
        self.assertEqual(self.shield.get_safe_actions([-0.999, -0.07]), [0, 1, 2])
        self.assertEqual(self.shield.get_safe_actions([-1.05, 0.0]), [0, 1, 2])
        self.assertEqual(self.shield.get_safe_actions([0.55, 0.01]), [0, 1, 2])

    def test_region_rule_only_applies_to_negative_velocity(self) -> None:
        self.assertEqual(self.shield.get_safe_actions([-1.05, -0.07]), [2])
        self.assertEqual(self.shield.get_safe_actions([-1.05, 0.0]), [0, 1, 2])
        self.assertEqual(self.shield.get_safe_actions([-1.05, 0.07]), [0, 1, 2])

    def test_one_step_prediction_matches_gymnasium(self) -> None:
        env = gym.make("MountainCar-v0").unwrapped
        try:
            state = np.asarray([-0.73, 0.015], dtype=float)
            env.state = state.copy()
            actual, *_ = env.step(0)
            np.testing.assert_allclose(
                self.shield.predict_next_state(state, 0), actual, atol=1e-7
            )
        finally:
            env.close()

    def test_deterministic_fallback_and_dataset_copy(self) -> None:
        state = np.asarray([-1.05, -0.01])
        self.assertEqual(self.shield.shield_action(state, 0), 2)
        collector = ShieldDatasetCollector(self.shield)
        sample = collector.append(state)
        state[0] = 0.0
        self.assertAlmostEqual(float(sample["state"][0]), -1.05)
        self.assertEqual(sample["safe_actions"], [2])


if __name__ == "__main__":
    unittest.main()
