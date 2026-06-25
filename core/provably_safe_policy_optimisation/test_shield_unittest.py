"""Unit tests for the runtime Shield."""

from __future__ import annotations

import unittest

import numpy as np
import torch as th

from provably_safe_policy_optimisation.shield import Shield, _default_obs_to_state


class ShieldTests(unittest.TestCase):
    def test_mask_validation(self) -> None:
        with self.assertRaises(ValueError):
            Shield(np.ones(4))  # 1-D

    def test_safe_action_passes_through(self) -> None:
        mask = np.array([[1, 0, 1]])
        sh = Shield(mask, seed=0)
        out = sh.override(np.array([0]), np.array([2]))  # action 2 is safe
        self.assertEqual(out.tolist(), [2])
        self.assertEqual(sh.diagnostics()["overridden"], 0)

    def test_unsafe_action_replaced_within_safe_set(self) -> None:
        mask = np.array([[1, 0, 1]])           # safe = {0, 2}
        sh = Shield(mask, seed=0)
        for _ in range(50):
            out = int(sh.override(np.array([0]), np.array([1]))[0])  # action 1 unsafe
            self.assertIn(out, (0, 2))
        self.assertEqual(sh.diagnostics()["overridden"], 50)

    def test_uniform_covers_all_safe_actions(self) -> None:
        mask = np.array([[1, 0, 1, 1]])        # safe = {0, 2, 3}
        sh = Shield(mask, seed=1)
        seen = {int(sh.override(np.array([0]), np.array([1]))[0]) for _ in range(200)}
        self.assertEqual(seen, {0, 2, 3})

    def test_no_safe_action_keep(self) -> None:
        mask = np.array([[0, 0]])
        sh = Shield(mask, no_safe_action="keep", seed=0)
        with self.assertWarns(UserWarning):
            out = sh.override(np.array([0]), np.array([1]))
        self.assertEqual(out.tolist(), [1])    # unchanged
        self.assertEqual(sh.diagnostics()["no_safe_state"], 1)

    def test_no_safe_action_raise(self) -> None:
        sh = Shield(np.array([[0, 0]]), no_safe_action="raise")
        with self.assertRaises(RuntimeError):
            sh.override(np.array([0]), np.array([0]))

    def test_batch_and_diagnostics(self) -> None:
        mask = np.array([[1, 0], [0, 1]])
        sh = Shield(mask, seed=0)
        out = sh.override(np.array([0, 1]), np.array([1, 1]))  # env0 unsafe, env1 safe
        self.assertEqual(out.tolist(), [0, 1])
        d = sh.diagnostics()
        self.assertEqual((d["checked"], d["overridden"]), (2, 1))
        self.assertAlmostEqual(d["intervention_rate"], 0.5)
        sh.reset_diagnostics()
        self.assertEqual(sh.diagnostics()["checked"], 0)

    def test_default_obs_to_state(self) -> None:
        self.assertEqual(_default_obs_to_state(np.array([3, 7])).tolist(), [3, 7])
        self.assertEqual(_default_obs_to_state(th.tensor([3, 7])).tolist(), [3, 7])

    def test_helpers(self) -> None:
        sh = Shield(np.array([[1, 0, 1]]))
        self.assertEqual(sh.safe_actions(0).tolist(), [0, 2])
        self.assertTrue(sh.is_safe(0, 2))
        self.assertFalse(sh.is_safe(0, 1))
        self.assertEqual((sh.n_states, sh.n_actions), (1, 3))


if __name__ == "__main__":
    unittest.main()
