"""Unit tests for success-rate early stopping support."""

from __future__ import annotations

import unittest

import torch

from experiments.pipelines.safety.frozenlake_slippery.core.config import get_pipeline_config
from experiments.pipelines.safety.frozenlake_slippery.core.env import make_env
from experiments.pipelines.safety.frozenlake_slippery.core.pipeline import (
    _downstream_ppo_config,
    _rashomon_ppo_config,
    _source_ppo_config,
)
from experiments.utils.ppo_utils import evaluate_with_success


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


class FrozenLakeSafetySuccessEarlyStoppingTests(unittest.TestCase):
    def test_env_defaults_to_slippery_and_reports_success_failure_rates(self) -> None:
        cfg = get_pipeline_config("diagonal_4x4")
        env = make_env(
            cfg.source_map,
            task_num=cfg.source_task_num,
            max_episode_steps=cfg.max_episode_steps,
            shaped=False,
        )
        actor = TableActor(
            {
                0: 2,
                1: 1,
                5: 1,
                9: 2,
                10: 2,
                11: 1,
            },
        )

        try:
            mean_reward, _, failure_rate, success_rate = evaluate_with_success(
                env,
                actor,
                episodes=1,
                deterministic=True,
                device="cpu",
            )
        finally:
            env.close()

        self.assertTrue(cfg.is_slippery)
        self.assertLessEqual(mean_reward, 1.0)
        self.assertGreaterEqual(failure_rate, 0.0)
        self.assertLessEqual(failure_rate, 1.0)
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)

    def test_ppo_configs_use_stochastic_success_rate_early_stopping(self) -> None:
        cfg = get_pipeline_config("diagonal_4x4")
        source_cfg = _source_ppo_config(
            cfg,
            seed=0,
            device="cpu",
            total_timesteps=cfg.source_total_timesteps,
        )
        downstream_cfg = _downstream_ppo_config(
            cfg,
            seed=0,
            device="cpu",
            total_timesteps=cfg.downstream_total_timesteps,
        )
        rashomon_cfg = _rashomon_ppo_config(
            cfg,
            seed=0,
            device="cpu",
            total_timesteps=cfg.rashomon_total_timesteps,
        )

        for ppo_cfg in (source_cfg, downstream_cfg, rashomon_cfg):
            with self.subTest(ppo_cfg=ppo_cfg):
                self.assertEqual(ppo_cfg.eval_episodes, 100)
                self.assertEqual(ppo_cfg.early_stop_min_steps, 5)
                self.assertIsNone(ppo_cfg.early_stop_reward_threshold)
                self.assertEqual(ppo_cfg.early_stop_failure_rate_threshold, 0.2)
                self.assertEqual(ppo_cfg.early_stop_success_rate_threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
