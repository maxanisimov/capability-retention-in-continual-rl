"""Unit tests for FrozenLake safety diagonal_4x4 settings."""

from __future__ import annotations

import unittest

from experiments.pipelines.behaviour_retention.frozenlake_safety_constrained.core.config import get_pipeline_config
from experiments.pipelines.behaviour_retention.frozenlake_safety_constrained.core.pipeline import (
    _downstream_ppo_config,
    _ewc_ppo_config,
    _rashomon_ppo_config,
    _source_ppo_config,
)
from experiments.pipelines.behaviour_retention.frozenlake_safety_constrained.core.reference_settings import (
    LAYOUT,
    frozenlake_safety_diagonal_4x4_settings,
)


PPO_FIELDS = (
    "total_timesteps",
    "eval_episodes",
    "rollout_steps",
    "update_epochs",
    "minibatch_size",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "ent_coef",
    "vf_coef",
    "lr",
    "max_grad_norm",
)


class FrozenLakeSafetyReferenceSettingsTests(unittest.TestCase):
    def assert_ppo_matches(self, ppo_cfg, expected: dict) -> None:
        for field_name in PPO_FIELDS:
            self.assertEqual(getattr(ppo_cfg, field_name), expected[field_name], field_name)

    def test_source_ppo_matches_diagonal_4x4_settings(self) -> None:
        cfg = get_pipeline_config("diagonal_4x4")
        reference = frozenlake_safety_diagonal_4x4_settings()
        expected = dict(reference["source"]["ppo"])
        expected["eval_episodes"] = reference["source"]["raw_eval"]["episodes"]
        ppo_cfg = _source_ppo_config(
            cfg,
            seed=3,
            device="cpu",
            total_timesteps=cfg.source_total_timesteps,
        )

        self.assertEqual(cfg.reference_layout, LAYOUT)
        self.assertEqual(cfg.hidden, expected["hidden"])
        self.assertEqual(cfg.activation, reference["source"]["activation"])
        self.assert_ppo_matches(ppo_cfg, expected)
        self.assertEqual(ppo_cfg.early_stop_reward_threshold, expected["early_stop_reward_threshold"])
        self.assertEqual(ppo_cfg.early_stop_failure_rate_threshold, expected["early_stop_failure_rate_threshold"])
        self.assertIsNone(ppo_cfg.early_stop_success_rate_threshold)

    def test_downstream_ppo_matches_diagonal_4x4_settings_for_all_adapters(self) -> None:
        cfg = get_pipeline_config("diagonal_4x4")
        expected = frozenlake_safety_diagonal_4x4_settings()["adaptation_ppo"]["ppo"]

        for ppo_cfg in (
            _downstream_ppo_config(
                cfg,
                seed=3,
                device="cpu",
                total_timesteps=cfg.downstream_total_timesteps,
            ),
            _rashomon_ppo_config(
                cfg,
                seed=3,
                device="cpu",
                total_timesteps=cfg.rashomon_total_timesteps,
            ),
        ):
            with self.subTest(ppo_cfg=ppo_cfg):
                self.assert_ppo_matches(ppo_cfg, expected)
                self.assertEqual(
                    ppo_cfg.early_stop_reward_threshold,
                    expected["early_stop_deterministic_total_reward_threshold"],
                )
                self.assertIsNone(ppo_cfg.early_stop_failure_rate_threshold)
                self.assertIsNone(ppo_cfg.early_stop_success_rate_threshold)

    def test_ewc_and_rashomon_settings_match_diagonal_4x4(self) -> None:
        cfg = get_pipeline_config("diagonal_4x4")
        reference = frozenlake_safety_diagonal_4x4_settings()
        expected_ewc = reference["adaptation_ewc"]["ewc"]
        expected_rashomon = reference["adaptation_rashomon"]["rashomon"]
        ppo_cfg = _ewc_ppo_config(
            cfg,
            seed=3,
            device="cpu",
            total_timesteps=cfg.downstream_total_timesteps,
            ewc_lambda=cfg.ewc_lambda,
        )

        self.assertEqual(ppo_cfg.ewc_lambda, expected_ewc["ewc_lambda"])
        self.assertEqual(ppo_cfg.ewc_apply_to_critic, expected_ewc["ewc_apply_to_critic"])
        self.assertEqual(cfg.fisher_sample_size, 10_000)
        self.assertEqual(cfg.rashomon_n_iters, expected_rashomon["rashomon_n_iters"])
        self.assertEqual(cfg.inverse_temp_start, expected_rashomon["inverse_temp_start"])
        self.assertEqual(cfg.inverse_temp_max, expected_rashomon["inverse_temp_max"])
        self.assertEqual(cfg.rashomon_checkpoint, expected_rashomon["rashomon_checkpoint"])
        self.assertEqual(cfg.rashomon_surrogate_aggregation, "min")
        self.assertEqual(cfg.rashomon_min_hard_spec, 1.0)


if __name__ == "__main__":
    unittest.main()
