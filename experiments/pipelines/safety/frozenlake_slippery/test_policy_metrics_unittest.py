"""Unit tests for post-training FrozenLake slippery shield safety policy metrics."""

from __future__ import annotations

import unittest

import torch

from experiments.pipelines.safety.frozenlake_slippery.core.config import get_pipeline_config
from experiments.pipelines.safety.frozenlake_slippery.core.pipeline import _compute_task_policy_metrics


class TableActor(torch.nn.Module):
    def __init__(self, actions_by_state: dict[int, int]):
        super().__init__()
        self.actions_by_state = dict(actions_by_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
        for idx, obs in enumerate(x.detach().cpu()):
            row = int(round(float(obs[0].item()) * 3))
            col = int(round(float(obs[1].item()) * 3))
            state_index = row * 4 + col
            action = self.actions_by_state.get(state_index, 0)
            logits[idx, action] = 10.0
        return logits


class FrozenLakeSafetyPolicyMetricTests(unittest.TestCase):
    def test_source_task_metrics_include_critical_safety_trajectory_safety_and_reward(self) -> None:
        cfg = get_pipeline_config("diagonal_4x4")
        actor = TableActor(
            {
                0: 2,
                1: 1,
                4: 0,
                5: 1,
                6: 0,
                9: 2,
                10: 2,
                11: 1,
                13: 2,
            },
        )

        metrics = _compute_task_policy_metrics(
            cfg,
            actor=actor,
            task="source",
            seed=0,
            device="cpu",
        )

        self.assertGreater(metrics["safety_critical_state_count"], 0)
        self.assertGreaterEqual(metrics["safety_critical_state_safe_count"], 0)
        self.assertLessEqual(
            metrics["safety_critical_state_safe_count"],
            metrics["safety_critical_state_count"],
        )
        self.assertGreaterEqual(metrics["safety_critical_state_safety_rate"], 0.0)
        self.assertLessEqual(metrics["safety_critical_state_safety_rate"], 1.0)
        self.assertIn(metrics["greedy_trajectory_safety"], {0.0, 1.0})
        self.assertIsInstance(metrics["total_reward"], float)


if __name__ == "__main__":
    unittest.main()
