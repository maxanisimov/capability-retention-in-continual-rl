"""Unit tests for FrozenLake shield safety diagonal_4x4 settings."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from experiments.pipelines.safety.frozenlake.core.config import get_pipeline_config
from experiments.pipelines.safety.frozenlake.core.pipeline import (
    adapt_downstream,
    _downstream_ppo_config,
    _ewc_ppo_config,
    _load_source_shield_metadata,
    main as pipeline_main,
    _rashomon_ppo_config,
    _source_ppo_config,
)
from experiments.pipelines.safety.frozenlake.core.reference_settings import (
    LAYOUT,
    frozenlake_shield_safety_diagonal_4x4_settings,
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
        reference = frozenlake_shield_safety_diagonal_4x4_settings()
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
        expected = frozenlake_shield_safety_diagonal_4x4_settings()["adaptation_ppo"]["ppo"]

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
        reference = frozenlake_shield_safety_diagonal_4x4_settings()
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

    def test_shield_settings_match_diagonal_4x4(self) -> None:
        cfg = get_pipeline_config("diagonal_4x4")
        reference = frozenlake_shield_safety_diagonal_4x4_settings()
        expected = reference["shield"]

        self.assertEqual(cfg.shield_type, expected["shield_type"])
        self.assertEqual(cfg.shield_risk_threshold, expected["shield_risk_threshold"])
        self.assertEqual(cfg.shield_theta, expected["shield_theta"])
        self.assertEqual(cfg.shield_max_vi_steps, expected["shield_max_vi_steps"])
        self.assertEqual(cfg.unsafe_cost_threshold, expected["unsafe_cost_threshold"])
        self.assertIn("shield", cfg.reference_settings_files)

    def test_source_dry_run_reports_shield_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = pipeline_main(
                    [
                        "--mode",
                        "source",
                        "--outputs-root",
                        tmp_dir,
                        "--shield-type",
                        "probabilistic",
                        "--shield-risk-threshold",
                        "0.2",
                        "--shield-theta",
                        "1e-8",
                        "--shield-max-vi-steps",
                        "50",
                        "--unsafe-cost-threshold",
                        "0.75",
                        "--dry-run",
                    ],
                )

        self.assertEqual(rc, 0)
        output = stdout.getvalue()
        self.assertIn("shield_type=probabilistic", output)
        self.assertIn("shield_risk_threshold=0.2", output)
        self.assertIn("shield_theta=1e-08", output)
        self.assertIn("shield_max_vi_steps=50", output)
        self.assertIn("unsafe_cost_threshold=0.75", output)

    def test_rashomon_adaptation_reports_missing_source_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "source"
            source_dir.mkdir()
            args = argparse.Namespace(
                layout="diagonal_4x4",
                seed=0,
                outputs_root=Path(tmp_dir),
                source_run_dir=source_dir,
            )

            with self.assertRaisesRegex(FileNotFoundError, "Source Rashomon dataset not found"):
                adapt_downstream(args, mode="downstream_rashomon")

    def test_source_shield_metadata_includes_min_risk_dataset_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir)
            (source_dir / "run_summary.yaml").write_text(
                "\n".join(
                    [
                        "run_settings:",
                        "  dataset_source: synthesized_shield",
                        "  shield_dataset_generation_mode: probabilistic_min_risk",
                        "  dataset_allowed_action_risk_count: 7",
                        "  dataset_allowed_action_risk_min: 0.0",
                        "  dataset_allowed_action_risk_max: 0.05",
                        "  dataset_allowed_action_risk_mean: 0.01",
                    ],
                ),
                encoding="utf-8",
            )

            metadata = _load_source_shield_metadata(source_dir)

        self.assertEqual(metadata["source_shield_dataset_generation_mode"], "probabilistic_min_risk")
        self.assertEqual(metadata["source_dataset_allowed_action_risk_count"], 7)
        self.assertEqual(metadata["source_dataset_allowed_action_risk_max"], 0.05)


if __name__ == "__main__":
    unittest.main()
