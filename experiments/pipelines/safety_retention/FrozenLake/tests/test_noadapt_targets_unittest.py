"""Unit tests for NoAdapt supervised target construction."""

from __future__ import annotations

import unittest

import torch

from experiments.pipelines.safety_retention.FrozenLake.core.config import SOURCE_MAP
from experiments.pipelines.safety_retention.FrozenLake.core.env import obs_to_state_index
from experiments.pipelines.safety_retention.FrozenLake.core.safety import (
    TrajectoryStep,
    build_noadapt_supervised_payload,
    create_rashomon_dataset,
)


class FrozenLakeNoAdaptTargetTests(unittest.TestCase):
    def test_trajectory_states_are_one_hot_and_off_trajectory_stays_multihot(self) -> None:
        rashomon_payload = create_rashomon_dataset(SOURCE_MAP, task_num=0.0)
        trajectory = [
            TrajectoryStep(step=0, state_index=0, row=0, col=0, action=2),
            TrajectoryStep(step=1, state_index=1, row=0, col=1, action=1),
        ]

        supervised = build_noadapt_supervised_payload(
            rashomon_payload,
            env_map=SOURCE_MAP,
            trajectory_steps=trajectory,
        )
        masks_by_state = {
            obs_to_state_index(obs.numpy(), SOURCE_MAP): actions
            for obs, actions in zip(supervised["state"], supervised["actions"], strict=True)
        }

        torch.testing.assert_close(masks_by_state[0], torch.tensor([0.0, 0.0, 1.0, 0.0]))
        torch.testing.assert_close(masks_by_state[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
        torch.testing.assert_close(masks_by_state[6], torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def test_unsafe_trajectory_action_is_rejected(self) -> None:
        rashomon_payload = create_rashomon_dataset(SOURCE_MAP, task_num=0.0)
        trajectory = [
            TrajectoryStep(step=0, state_index=1, row=0, col=1, action=2),
        ]

        with self.assertRaises(ValueError):
            build_noadapt_supervised_payload(
                rashomon_payload,
                env_map=SOURCE_MAP,
                trajectory_steps=trajectory,
            )


if __name__ == "__main__":
    unittest.main()

