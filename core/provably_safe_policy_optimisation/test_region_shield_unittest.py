"""Unit tests for RegionShield (continuous states, discrete actions)."""

from __future__ import annotations

import unittest

import numpy as np
import torch as th

from provably_safe_policy_optimisation.shield import RegionShield


class RegionShieldTests(unittest.TestCase):
    def test_classify_and_fallback(self) -> None:
        # region 0: x < 0 -> {2}; region 1: x > 1 -> {0}; else fallback (all safe).
        rs = RegionShield(
            regions=[(lambda o: o[0] < 0.0, [2]), (lambda o: o[0] > 1.0, [0])],
            n_actions=3,
            seed=0,
        )
        states = rs.obs_to_state(np.array([[-0.5, 0.0], [2.0, 0.0], [0.5, 0.0]]))
        self.assertEqual(states.tolist(), [0, 1, 2])  # 2 == fallback index
        # fallback row is all-safe by default
        self.assertEqual(rs.mask[2].tolist(), [1, 1, 1])

    def test_first_match_wins_on_overlap(self) -> None:
        rs = RegionShield(
            regions=[(lambda o: o[0] < 5.0, [0]), (lambda o: o[0] < 10.0, [1])],
            n_actions=2,
            seed=0,
        )
        self.assertEqual(int(rs.obs_to_state(np.array([[3.0]]))[0]), 0)  # both match -> region 0

    def test_override_uses_region_safe_actions(self) -> None:
        rs = RegionShield(regions=[(lambda o: o[0] < 0.0, [2])], n_actions=3, seed=0)
        states = rs.obs_to_state(np.array([[-0.5, 0.0]]))
        out = rs.override(states, np.array([0]))  # action 0 unsafe in region 0
        self.assertEqual(out.tolist(), [2])
        self.assertEqual(rs.diagnostics()["overridden"], 1)

    def test_default_safe_actions_and_raise(self) -> None:
        rs = RegionShield(regions=[(lambda o: o[0] < 0.0, [2])], n_actions=3,
                          default_safe_actions=[1], seed=0)
        self.assertEqual(rs.mask[rs._fallback_index].tolist(), [0, 1, 0])
        out = rs.override(rs.obs_to_state(np.array([[5.0, 0.0]])), np.array([0]))  # fallback
        self.assertEqual(out.tolist(), [1])

    def test_invalid_safe_action_raises(self) -> None:
        with self.assertRaises(ValueError):
            RegionShield(regions=[(lambda o: True, [5])], n_actions=3)

    def test_torch_obs_accepted(self) -> None:
        rs = RegionShield(regions=[(lambda o: o[0] < 0.0, [2])], n_actions=3, seed=0)
        states = rs.obs_to_state(th.tensor([[-0.5, 0.0], [0.5, 0.0]]))
        self.assertEqual(states.tolist(), [0, 1])

    def test_from_boxes(self) -> None:
        # x <= 0 -> {2}; x >= 1 -> {0}.
        rs = RegionShield.from_boxes(
            [([-np.inf, -np.inf], [0.0, np.inf], [2]),
             ([1.0, -np.inf], [np.inf, np.inf], [0])],
            n_actions=3, seed=0,
        )
        states = rs.obs_to_state(np.array([[-1.0, 0.0], [2.0, 0.0], [0.5, 0.0]]))
        self.assertEqual(states.tolist(), [0, 1, 2])
        self.assertEqual(rs.override(np.array([0]), np.array([0])).tolist(), [2])

    def test_from_boxes_validates_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RegionShield.from_boxes([([1.0], [0.0], [0])], n_actions=2)  # low > high


if __name__ == "__main__":
    unittest.main()
