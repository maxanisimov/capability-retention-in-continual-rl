"""Tests for the heuristic LunarLander continuous-state shield."""

from __future__ import annotations

import unittest

from continuous_state_shields import LunarLanderShield


class LunarLanderShieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shield = LunarLanderShield()

    def test_nominal_high_altitude_state_allows_all_actions(self) -> None:
        state = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(self.shield.get_safe_actions(state), [0, 1, 2, 3])

    def test_dangerous_low_descent_requires_main_engine(self) -> None:
        state = [0.0, 0.3, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(self.shield.get_safe_actions(state), [2])

    def test_horizontal_boundaries_reject_outward_side_engine(self) -> None:
        right = [0.85, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        left = [-0.85, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertNotIn(1, self.shield.get_safe_actions(right))
        self.assertNotIn(3, self.shield.get_safe_actions(left))

    def test_excessive_tilt_rejects_worsening_engine(self) -> None:
        positive_tilt = [0.0, 0.4, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]
        negative_tilt = [0.0, 0.4, 0.0, 0.0, -0.4, 0.0, 0.0, 0.0]
        self.assertNotIn(3, self.shield.get_safe_actions(positive_tilt))
        self.assertNotIn(1, self.shield.get_safe_actions(negative_tilt))

    def test_conflict_uses_vertical_descent_priority(self) -> None:
        # At the right edge and tilted left, the heuristic marks main thrust as
        # horizontally worsening, but dangerous descent has explicit priority.
        state = [0.9, 0.3, 0.0, -0.6, -0.5, 0.0, 0.0, 0.0]
        safe, info = self.shield.get_safe_actions(state, return_info=True)
        self.assertEqual(safe, [2])
        self.assertTrue(info["priority_fallback"])
        self.assertIn("vertical-descent priority", info["reason"])

    def test_unsafe_action_is_overridden_deterministically(self) -> None:
        state = [0.0, 0.3, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(self.shield.shield_action(state, 0), 2)
        self.assertEqual(self.shield.shield_action(state, 2), 2)


if __name__ == "__main__":
    unittest.main()
