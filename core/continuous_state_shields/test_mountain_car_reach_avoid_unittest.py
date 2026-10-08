"""Tests for the synthesized MountainCar reach-avoid shield."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np

from continuous_state_shields import (
    MountainCarIntervalBoxShield,
    MountainCarReachAvoidShield,
    box_reach_avoid_grid,
    save_interval_box_shield,
    save_reach_avoid_grid,
    synthesise_reach_avoid_grid,
)


class MountainCarReachAvoidShieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = synthesise_reach_avoid_grid(
            n_positions=73,
            n_velocities=57,
            goal_horizon=200,
            boundary_margin=0.025,
        )

    def _shield(self) -> MountainCarReachAvoidShield:
        return MountainCarReachAvoidShield(
            positions=self.payload["positions"],
            velocities=self.payload["velocities"],
            winning=self.payload["winning"],
            goal_rank=self.payload["goal_rank"],
            action_mask=self.payload["action_mask"],
            metadata=self.payload["metadata"],
        )

    def test_artifact_round_trip_and_nonempty_initial_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shield.npz"
            save_reach_avoid_grid(path, self.payload)
            shield = MountainCarReachAvoidShield.load(path)
        safe = shield.get_safe_actions([-0.5, 0.0])
        self.assertTrue(safe)
        self.assertIn(shield.preferred_action([-0.5, 0.0]), safe)

    def test_boundary_fails_closed_and_goal_is_unconstrained(self) -> None:
        shield = self._shield()
        self.assertEqual(shield.get_safe_actions([-1.2, -0.01]), [2])
        self.assertEqual(shield.get_safe_actions([0.5, 0.01]), [0, 1, 2])

    def test_transition_matches_gymnasium(self) -> None:
        shield = self._shield()
        env = gym.make("MountainCar-v0").unwrapped
        try:
            state = np.asarray([-0.73, 0.015], dtype=float)
            env.state = state.copy()
            actual, *_ = env.step(0)
            np.testing.assert_allclose(
                shield.predict_next_state(state, 0), actual, atol=1e-7
            )
        finally:
            env.close()

    def test_box_conversion_round_trip_and_closed_boundary_intersection(self) -> None:
        boxed = box_reach_avoid_grid(
            positions=self.payload["positions"],
            velocities=self.payload["velocities"],
            action_mask=self.payload["action_mask"],
            velocity_bands=8,
            source_metadata=self.payload["metadata"],
        )
        self.assertGreater(len(boxed["box_lows"]), 0)
        self.assertEqual(boxed["metadata"]["velocity_bands"], 8)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "boxed.npz"
            save_interval_box_shield(path, boxed)
            shield = MountainCarIntervalBoxShield.load(path)
        self.assertEqual(len(shield.box_lows), len(boxed["box_lows"]))

        boundary_shield = MountainCarIntervalBoxShield(
            box_lows=np.asarray([[-1.0, -0.1], [-0.5, -0.1]]),
            box_highs=np.asarray([[-0.5, 0.1], [0.0, 0.1]]),
            safe_masks=np.asarray([[False, True, True], [False, False, True]]),
        )
        self.assertEqual(boundary_shield.get_safe_actions([-0.75, 0.0]), [1, 2])
        self.assertEqual(boundary_shield.get_safe_actions([-0.5, 0.0]), [2])
        self.assertEqual(boundary_shield.get_safe_actions([0.25, 0.0]), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
