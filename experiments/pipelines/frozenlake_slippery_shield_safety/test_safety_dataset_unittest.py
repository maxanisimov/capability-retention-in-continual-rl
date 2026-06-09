"""Unit tests for FrozenLake slippery shield safety Rashomon dataset construction."""

from __future__ import annotations

import unittest

import torch

from experiments.pipelines.frozenlake_slippery_shield_safety.core.config import SOURCE_MAP
from experiments.pipelines.frozenlake_slippery_shield_safety.core.env import obs_to_state_index
from experiments.pipelines.frozenlake_slippery_shield_safety.core.safety import (
    create_rashomon_dataset,
    create_shield_rashomon_dataset,
    frozenlake_cost_fn,
    frozenlake_label_fn,
    frozenlake_transition_matrix,
    min_risk_shield_from_action_risk,
    shield_allowed_action_risk_stats,
    synthesise_frozenlake_shield,
)


class FrozenLakeSafetyDatasetTests(unittest.TestCase):
    def test_transition_matrix_shape_and_probabilities(self) -> None:
        matrix = frozenlake_transition_matrix(SOURCE_MAP)

        self.assertEqual(matrix.shape, (16, 16, 4))
        self.assertTrue(torch.tensor(matrix).ge(0.0).all().item())
        torch.testing.assert_close(
            torch.tensor(matrix.sum(axis=0), dtype=torch.float64),
            torch.ones((16, 4), dtype=torch.float64),
        )
        self.assertGreater(int((matrix[:, 0, 2] > 0.0).sum()), 1)
        torch.testing.assert_close(torch.tensor(matrix[2, 2, :]), torch.ones(4, dtype=torch.float64))
        torch.testing.assert_close(torch.tensor(matrix[15, 15, :]), torch.ones(4, dtype=torch.float64))

    def test_transition_matrix_uses_configurable_success_rate(self) -> None:
        matrix = frozenlake_transition_matrix(SOURCE_MAP, success_rate=0.8)

        self.assertAlmostEqual(float(matrix[1, 0, 2]), 0.8)
        self.assertAlmostEqual(float(matrix[0, 0, 2]), 0.1)
        self.assertAlmostEqual(float(matrix[4, 0, 2]), 0.1)

    def test_holes_are_unsafe_labels(self) -> None:
        self.assertEqual(frozenlake_label_fn(SOURCE_MAP, 2), {"unsafe"})
        self.assertEqual(frozenlake_cost_fn(frozenlake_label_fn(SOURCE_MAP, 2)), 1.0)
        self.assertEqual(frozenlake_label_fn(SOURCE_MAP, 15), set())
        self.assertEqual(frozenlake_cost_fn(frozenlake_label_fn(SOURCE_MAP, 15)), 0.0)

    def test_probabilistic_shield_dataset_schema_and_representative_masks(self) -> None:
        shield, info = synthesise_frozenlake_shield(
            SOURCE_MAP,
            shield_type="probabilistic",
            risk_threshold=0.05,
        )
        payload = create_shield_rashomon_dataset(SOURCE_MAP, task_num=0.0, shield=shield)

        self.assertEqual(shield.shape, (16, 4))
        self.assertEqual(len(info.winning_states), 10)
        self.assertEqual(set(payload.keys()), {"state", "actions"})
        self.assertEqual(payload["state"].dtype, torch.float32)
        self.assertEqual(payload["actions"].dtype, torch.float32)
        self.assertEqual(tuple(payload["state"].shape), (9, 3))
        self.assertEqual(tuple(payload["actions"].shape), (9, 4))

        masks_by_state = {
            obs_to_state_index(obs.numpy(), SOURCE_MAP): actions
            for obs, actions in zip(payload["state"], payload["actions"], strict=True)
        }

        torch.testing.assert_close(masks_by_state[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
        torch.testing.assert_close(masks_by_state[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(masks_by_state[10], torch.tensor([0.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(masks_by_state[14], torch.tensor([1.0, 1.0, 1.0, 1.0]))

    def test_probabilistic_shield_records_action_risk_and_threshold_changes_masks(self) -> None:
        strict_shield, info = synthesise_frozenlake_shield(
            SOURCE_MAP,
            shield_type="probabilistic",
            risk_threshold=0.0,
        )
        loose_shield, _ = synthesise_frozenlake_shield(
            SOURCE_MAP,
            shield_type="probabilistic",
            risk_threshold=1.0,
        )

        self.assertEqual(strict_shield.shape, (16, 4))
        self.assertIsNotNone(info.action_risk)
        self.assertEqual(info.action_risk.shape, (16, 4))
        self.assertGreater(int(loose_shield.sum()), int(strict_shield.sum()))

    def test_min_risk_shield_includes_ties_within_theta(self) -> None:
        action_risk = torch.ones((16, 4), dtype=torch.float64).numpy()
        action_risk[0] = [0.2, 0.2 + 5e-11, 0.2 + 2e-10, 0.7]

        shield = min_risk_shield_from_action_risk(SOURCE_MAP, action_risk, theta=1e-10)

        torch.testing.assert_close(torch.tensor(shield[0]), torch.tensor([1, 1, 0, 0]))
        torch.testing.assert_close(torch.tensor(shield[2]), torch.tensor([0, 0, 0, 0]))
        torch.testing.assert_close(torch.tensor(shield[15]), torch.tensor([0, 0, 0, 0]))

    def test_min_risk_probabilistic_dataset_is_threshold_independent(self) -> None:
        payloads = []
        for threshold in (0.0, 1.0):
            _thresholded_shield, info = synthesise_frozenlake_shield(
                SOURCE_MAP,
                shield_type="probabilistic",
                risk_threshold=threshold,
                success_rate=0.8,
            )
            self.assertIsNotNone(info.action_risk)
            shield = min_risk_shield_from_action_risk(SOURCE_MAP, info.action_risk, theta=1e-10)
            payloads.append(create_shield_rashomon_dataset(SOURCE_MAP, task_num=0.0, shield=shield))

        torch.testing.assert_close(payloads[0]["state"], payloads[1]["state"])
        torch.testing.assert_close(payloads[0]["actions"], payloads[1]["actions"])

    def test_allowed_action_risk_stats_report_dataset_risks(self) -> None:
        _thresholded_shield, info = synthesise_frozenlake_shield(
            SOURCE_MAP,
            shield_type="probabilistic",
            risk_threshold=0.0,
            success_rate=0.8,
        )
        self.assertIsNotNone(info.action_risk)
        shield = min_risk_shield_from_action_risk(SOURCE_MAP, info.action_risk, theta=1e-10)

        stats = shield_allowed_action_risk_stats(SOURCE_MAP, shield, info.action_risk)

        self.assertGreater(stats["dataset_allowed_action_risk_count"], 0)
        self.assertLessEqual(
            stats["dataset_allowed_action_risk_min"],
            stats["dataset_allowed_action_risk_mean"],
        )
        self.assertLessEqual(
            stats["dataset_allowed_action_risk_mean"],
            stats["dataset_allowed_action_risk_max"],
        )

    def test_dataset_schema_and_representative_masks(self) -> None:
        payload = create_rashomon_dataset(SOURCE_MAP, task_num=0.0)

        self.assertEqual(set(payload.keys()), {"state", "actions"})
        self.assertEqual(payload["state"].dtype, torch.float32)
        self.assertEqual(payload["actions"].dtype, torch.float32)
        self.assertEqual(tuple(payload["state"].shape), (10, 3))
        self.assertEqual(tuple(payload["actions"].shape), (10, 4))

        masks_by_state = {
            obs_to_state_index(obs.numpy(), SOURCE_MAP): actions
            for obs, actions in zip(payload["state"], payload["actions"], strict=True)
        }

        torch.testing.assert_close(masks_by_state[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
        torch.testing.assert_close(masks_by_state[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(masks_by_state[4], torch.tensor([0.0, 0.0, 0.0, 1.0]))
        torch.testing.assert_close(masks_by_state[6], torch.tensor([1.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(masks_by_state[10], torch.tensor([0.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(masks_by_state[14], torch.tensor([1.0, 1.0, 1.0, 1.0]))

    def test_dataset_excludes_holes_and_goal(self) -> None:
        payload = create_rashomon_dataset(SOURCE_MAP, task_num=0.0)
        included = {
            obs_to_state_index(obs.numpy(), SOURCE_MAP)
            for obs in payload["state"]
        }
        holes_and_goal = {2, 3, 7, 8, 12, 15}

        self.assertTrue(included.isdisjoint(holes_and_goal))
        self.assertEqual(included, {0, 1, 4, 5, 6, 9, 10, 11, 13, 14})


if __name__ == "__main__":
    unittest.main()
