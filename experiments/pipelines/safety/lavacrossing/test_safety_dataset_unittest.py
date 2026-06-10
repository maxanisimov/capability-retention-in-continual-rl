"""Unit tests for LavaCrossing shield-safety dataset construction."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.pipelines.safety.lavacrossing.core.config import SOURCE_MAP
from experiments.pipelines.safety.lavacrossing.core.env import obs_to_state_index
from experiments.pipelines.safety.lavacrossing.core.safety import (
    create_shield_rashomon_dataset,
    lavacrossing_cost_fn,
    lavacrossing_label_fn,
    lavacrossing_transition_matrix,
    min_risk_shield_from_action_risk,
    shield_allowed_action_risk_stats,
    synthesise_lavacrossing_shield,
)


class LavaCrossingSafetyDatasetTests(unittest.TestCase):
    def test_transition_matrix_shape_probabilities_and_slip(self) -> None:
        matrix = lavacrossing_transition_matrix(SOURCE_MAP, slip_prob=0.1)

        self.assertEqual(matrix.shape, (49, 49, 5))
        self.assertTrue(np.all(matrix >= 0.0))
        self.assertTrue(np.allclose(matrix.sum(axis=0), 1.0))
        self.assertGreater(int((matrix[:, 0, 1] > 0.0).sum()), 1)

        lava_state = 7
        goal_state = 48
        self.assertTrue(np.allclose(matrix[lava_state, lava_state, :], 1.0))
        self.assertTrue(np.allclose(matrix[goal_state, goal_state, :], 1.0))

    def test_labels_and_costs(self) -> None:
        self.assertEqual(lavacrossing_label_fn(SOURCE_MAP, 0), set())
        self.assertEqual(lavacrossing_label_fn(SOURCE_MAP, 7), {"unsafe", "lava"})
        self.assertEqual(lavacrossing_label_fn(SOURCE_MAP, 48), {"goal"})
        self.assertEqual(lavacrossing_cost_fn(lavacrossing_label_fn(SOURCE_MAP, 7)), 1.0)
        self.assertEqual(lavacrossing_cost_fn(lavacrossing_label_fn(SOURCE_MAP, 48)), 0.0)

    def test_probabilistic_shield_records_action_risk_and_threshold_changes_masks(self) -> None:
        strict_shield, info = synthesise_lavacrossing_shield(
            SOURCE_MAP,
            shield_type="probabilistic",
            risk_threshold=0.0,
            slip_prob=0.1,
        )
        loose_shield, _ = synthesise_lavacrossing_shield(
            SOURCE_MAP,
            shield_type="probabilistic",
            risk_threshold=1.0,
            slip_prob=0.1,
        )

        self.assertEqual(strict_shield.shape, (49, 5))
        self.assertIsNotNone(info.action_risk)
        self.assertEqual(info.action_risk.shape, (49, 5))
        self.assertGreater(int(loose_shield.sum()), int(strict_shield.sum()))

    def test_min_risk_ties_include_actions_within_theta(self) -> None:
        action_risk = np.ones((49, 5), dtype=np.float64)
        action_risk[0] = [0.3, 0.3 + 5e-11, 0.3 + 2e-10, 0.8, 0.9]

        shield = min_risk_shield_from_action_risk(SOURCE_MAP, action_risk, theta=1e-10)

        torch.testing.assert_close(torch.tensor(shield[0]), torch.tensor([1, 1, 0, 0, 0]))
        torch.testing.assert_close(torch.tensor(shield[7]), torch.tensor([0, 0, 0, 0, 0]))
        torch.testing.assert_close(torch.tensor(shield[48]), torch.tensor([0, 0, 0, 0, 0]))

    def test_min_risk_dataset_is_threshold_independent(self) -> None:
        payloads = []
        for threshold in (0.0, 1.0):
            _thresholded, info = synthesise_lavacrossing_shield(
                SOURCE_MAP,
                shield_type="probabilistic",
                risk_threshold=threshold,
                slip_prob=0.1,
            )
            self.assertIsNotNone(info.action_risk)
            shield = min_risk_shield_from_action_risk(SOURCE_MAP, info.action_risk, theta=1e-10)
            payloads.append(create_shield_rashomon_dataset(SOURCE_MAP, task_num=0.0, shield=shield))

        torch.testing.assert_close(payloads[0]["state"], payloads[1]["state"])
        torch.testing.assert_close(payloads[0]["actions"], payloads[1]["actions"])
        self.assertEqual(payloads[0]["actions"].shape[1], 5)

    def test_dataset_excludes_lava_walls_and_goals(self) -> None:
        _thresholded, info = synthesise_lavacrossing_shield(
            SOURCE_MAP,
            shield_type="probabilistic",
            risk_threshold=0.0,
            slip_prob=0.1,
        )
        self.assertIsNotNone(info.action_risk)
        shield = min_risk_shield_from_action_risk(SOURCE_MAP, info.action_risk, theta=1e-10)
        payload = create_shield_rashomon_dataset(SOURCE_MAP, task_num=0.0, shield=shield)
        included = {obs_to_state_index(obs.numpy(), SOURCE_MAP) for obs in payload["state"]}
        excluded = {
            idx
            for idx, cell in enumerate("".join(SOURCE_MAP))
            if cell in {"L", "W", "G"}
        }

        self.assertTrue(included.isdisjoint(excluded))
        self.assertGreater(len(included), 0)

    def test_allowed_action_risk_stats_report_dataset_risks(self) -> None:
        _thresholded, info = synthesise_lavacrossing_shield(
            SOURCE_MAP,
            shield_type="probabilistic",
            risk_threshold=0.0,
            slip_prob=0.1,
        )
        self.assertIsNotNone(info.action_risk)
        shield = min_risk_shield_from_action_risk(SOURCE_MAP, info.action_risk, theta=1e-10)

        stats = shield_allowed_action_risk_stats(SOURCE_MAP, shield, info.action_risk)

        self.assertGreater(stats["dataset_allowed_action_risk_count"], 0)
        self.assertLessEqual(stats["dataset_allowed_action_risk_min"], stats["dataset_allowed_action_risk_mean"])
        self.assertLessEqual(stats["dataset_allowed_action_risk_mean"], stats["dataset_allowed_action_risk_max"])


if __name__ == "__main__":
    unittest.main()
