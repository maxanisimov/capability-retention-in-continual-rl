"""Tests for the descent-only LunarLander shield."""

from __future__ import annotations

import unittest

import numpy as np

from continuous_state_shields import (
    LunarLanderDescentShield,
    LunarLanderDescentShieldConfig,
)


def _state(altitude: float, vertical_speed: float) -> np.ndarray:
    return np.array(
        [0.1, altitude, -0.2, vertical_speed, 0.05, -0.1, 0.0, 0.0], dtype=float
    )


class LunarLanderDescentShieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shield = LunarLanderDescentShield()

    def test_fast_descent_near_ground_admits_only_the_main_engine(self) -> None:
        self.assertEqual(self.shield.get_safe_actions(_state(0.4, -0.5)), [2])

    def test_high_altitude_is_unconstrained(self) -> None:
        self.assertEqual(self.shield.get_safe_actions(_state(1.2, -0.9)), [0, 1, 2, 3])

    def test_slow_descent_is_unconstrained(self) -> None:
        self.assertEqual(self.shield.get_safe_actions(_state(0.2, -0.1)), [0, 1, 2, 3])

    def test_ascent_near_the_ground_is_unconstrained(self) -> None:
        self.assertEqual(self.shield.get_safe_actions(_state(0.2, 0.3)), [0, 1, 2, 3])

    def test_band_is_open_at_its_boundary(self) -> None:
        config = self.shield.config
        boundary = _state(config.critical_height, config.safe_min_vertical_speed)
        self.assertEqual(self.shield.get_safe_actions(boundary), [0, 1, 2, 3])
        self.assertFalse(self.shield.in_critical_band(boundary))

    def test_band_sits_outside_the_unsafe_set_it_protects(self) -> None:
        # The shield must engage before a violation, not on top of one: a state
        # that is about to violate y < 0.5 and v_y < -0.4 is already shielded.
        config = self.shield.config
        self.assertGreater(config.critical_height, 0.5)
        self.assertGreater(config.safe_min_vertical_speed, -0.4)
        self.assertTrue(self.shield.in_critical_band(_state(0.55, -0.38)))

    def test_unsafe_proposal_is_overridden_to_the_main_engine(self) -> None:
        self.assertEqual(self.shield.shield_action(_state(0.4, -0.5), 0), 2)
        self.assertEqual(self.shield.shield_action(_state(0.4, -0.5), 2), 2)

    def test_safe_proposal_passes_through_outside_the_band(self) -> None:
        self.assertEqual(self.shield.shield_action(_state(1.2, -0.1), 3), 3)

    def test_diagnostics_report_the_active_component(self) -> None:
        _actions, info = self.shield.get_safe_actions(
            _state(0.4, -0.5), return_info=True
        )
        self.assertEqual(info["violated_constraints"], ["vertical_descent"])
        self.assertTrue(info["shield_active"])

    def test_wrong_observation_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"shape \(8,\)"):
            self.shield.get_safe_actions(np.zeros(6))

    def test_config_rejects_a_non_negative_descent_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be negative"):
            LunarLanderDescentShieldConfig(safe_min_vertical_speed=0.1)

    def test_config_rejects_an_out_of_range_main_engine_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "main_engine_action"):
            LunarLanderDescentShieldConfig(main_engine_action=4)


if __name__ == "__main__":
    unittest.main()
