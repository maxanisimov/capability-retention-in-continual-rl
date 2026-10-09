"""Tests for the CartPole continuous-state shield."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from continuous_state_shields import CartPoleShield


class CartPoleShieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shield = CartPoleShield()

    def test_central_upright_state_allows_both_actions(self) -> None:
        self.assertEqual(self.shield.get_safe_actions([0.0, 0.0, 0.0, 0.0]), [0, 1])

    def test_right_boundary_rejects_right_push(self) -> None:
        self.assertEqual(self.shield.get_safe_actions([2.19, 0.5, 0.0, 0.0]), [0])

    def test_left_boundary_rejects_left_push(self) -> None:
        self.assertEqual(self.shield.get_safe_actions([-2.19, -0.5, 0.0, 0.0]), [1])

    def test_positive_and_negative_pole_angles_choose_corrective_push(self) -> None:
        self.assertEqual(self.shield.get_safe_actions([0.0, 0.0, 0.19, 0.0]), [1])
        self.assertEqual(self.shield.get_safe_actions([0.0, 0.0, -0.19, 0.0]), [0])

    def test_safe_set_is_never_empty_over_random_states(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(1_000):
            state = rng.uniform([-2.4, -4.0, -0.21, -4.0], [2.4, 4.0, 0.21, 4.0])
            self.assertTrue(self.shield.get_safe_actions(state))

    def test_single_step_prediction_matches_gymnasium(self) -> None:
        env = gym.make("CartPole-v1").unwrapped
        try:
            state = np.asarray([0.1, -0.2, 0.03, 0.04], dtype=float)
            env.state = state.copy()
            actual, *_ = env.step(1)
            np.testing.assert_allclose(
                self.shield.predict_next_state(state, 1, steps=1),
                actual,
                rtol=1e-6,
                atol=1e-7,
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
