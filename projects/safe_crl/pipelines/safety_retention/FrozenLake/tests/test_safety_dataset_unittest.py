"""Unit tests for FrozenLake safety Rashomon dataset construction."""

from __future__ import annotations

import unittest

import torch

from projects.safe_crl.pipelines.safety_retention.frozenlake.core.config import SOURCE_MAP
from projects.safe_crl.pipelines.safety_retention.frozenlake.core.env import obs_to_state_index
from projects.safe_crl.pipelines.safety_retention.frozenlake.core.safety import create_rashomon_dataset


class FrozenLakeSafetyDatasetTests(unittest.TestCase):
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
        torch.testing.assert_close(masks_by_state[1], torch.tensor([1.0, 1.0, 0.0, 1.0]))
        torch.testing.assert_close(masks_by_state[6], torch.tensor([1.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(masks_by_state[11], torch.tensor([1.0, 1.0, 1.0, 0.0]))

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

