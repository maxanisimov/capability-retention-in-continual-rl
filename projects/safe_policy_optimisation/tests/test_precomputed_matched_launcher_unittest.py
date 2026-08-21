"""Tests for compute-matched precomputed PSPO support."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from projects.safe_policy_optimisation.scripts.run_precomputed_matched_adaptive_budgets import (
    BasePolicyMetadata,
    build_safe_set_command,
    build_train_command,
    load_adaptive_seed_spec,
    parse_cpu_ids,
    validate_shared_settings,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    build_base_policy,
    load_base_policy_for_dataset,
)


def _adaptive_config(seed: int, base_policy_path: Path) -> dict:
    return {
        "algorithm": "adaptive_safe_ppo_v2",
        "seed": seed,
        "base_policy_path": str(base_policy_path),
        "base_policy_architecture": {
            "input_dim": 2,
            "n_actions": 2,
            "hidden_dim": 64,
            "n_hidden": 0,
            "activation": "Tanh",
            "state_representation": "one_hot_discrete_observation",
        },
        "shield_path": "shield.pt",
        "env_id": "TestEnv-v0",
        "env_kwargs": {"example": 1},
        "max_episode_steps": 40,
        "cost_limit": 0.001,
        "total_timesteps": 25_000,
        "eval_episodes": 100,
        "evaluation_policy": "unshielded",
        "early_stop_eval_policy": "unshielded",
        "early_stop_eval_freq": 0,
        "early_stop_eval_episodes": 100,
        "early_stop_success_rate": 1.0,
        "success_reward_threshold": -0.5,
        "curve_eval_freq": 2048,
        "curve_eval_episodes": 20,
        "shield_key": "shield",
        "shield_source": "shield",
        "shield_action_storage": "proposed",
        "risk_threshold": None,
        "training_hyperparameters": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "adaptive": {
            "rashomon_checkpoint": 100,
            "rashomon_batch_size": 500,
            "certificate_samples": 1000,
            "rashomon_multi_label_mode": "any",
            "rashomon_surrogate": "auto",
            "safe_region_shape": "orthotope",
        },
    }


class ExistingBasePolicyTests(unittest.TestCase):
    def test_loads_exact_existing_policy_and_revalidates_margin(self) -> None:
        dataset = {
            "state": torch.eye(2),
            "actions": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        metadata = {"state_representation": "one_hot_discrete_observation"}
        original = build_base_policy(2, 2, hidden_dim=64, n_hidden=0)
        with torch.no_grad():
            original[0].weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
            original[0].bias.zero_()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base_policy.pt"
            torch.save(
                {
                    "architecture": {
                        "input_dim": 2,
                        "n_actions": 2,
                        "hidden_dim": 64,
                        "n_hidden": 0,
                        "activation": "Tanh",
                        "state_representation": "one_hot_discrete_observation",
                    },
                    "state_dict": original.state_dict(),
                },
                path,
            )
            loaded, metrics, source = load_base_policy_for_dataset(
                path,
                dataset,
                metadata,
                hidden_dim=64,
                n_hidden=0,
                target_margin=2.0,
                margin_mode="any",
                device="cpu",
            )

            for expected, actual in zip(original.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(metrics["reached_target"])
            self.assertEqual(metrics["final_min_margin"], 2.0)
            self.assertEqual(source["path"], str(path.resolve()))
            self.assertEqual(len(source["sha256"]), 64)

    def test_rejects_incompatible_existing_policy(self) -> None:
        dataset = {
            "state": torch.eye(2),
            "actions": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        metadata = {"state_representation": "decoded_features"}
        model = build_base_policy(2, 2, hidden_dim=64, n_hidden=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base_policy.pt"
            torch.save(
                {
                    "architecture": {
                        "input_dim": 2,
                        "n_actions": 2,
                        "hidden_dim": 64,
                        "n_hidden": 0,
                        "activation": "Tanh",
                        "state_representation": "one_hot_discrete_observation",
                    },
                    "state_dict": model.state_dict(),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "state_representation"):
                load_base_policy_for_dataset(
                    path,
                    dataset,
                    metadata,
                    hidden_dim=64,
                    n_hidden=0,
                    target_margin=0.1,
                    margin_mode="any",
                    device="cpu",
                )


class MatchedBudgetLauncherTests(unittest.TestCase):
    def test_cpu_parser_accepts_ranges(self) -> None:
        self.assertEqual(parse_cpu_ids("31-33,40,32"), [31, 32, 33, 40])

    def test_extracts_exact_budget_and_builds_seed_specific_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = root / "base_policy.pt"
            base_path.write_bytes(b"base")
            adaptive = root / "adaptive"
            seed_dir = adaptive / "seed7"
            seed_dir.mkdir(parents=True)
            config = _adaptive_config(7, base_path)
            (seed_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (seed_dir / "summary.json").write_text(
                json.dumps({"adaptive_diagnostics": {"rashomon_iters_spent": 318_490}}),
                encoding="utf-8",
            )

            spec = load_adaptive_seed_spec(adaptive, 7)
            validate_shared_settings([spec])
            metadata = BasePolicyMetadata(
                path=base_path,
                sha256="abc",
                architecture=config["base_policy_architecture"],
                target_margin=2.0,
                margin_mode="any",
            )
            output = root / "out"
            safe_command = build_safe_set_command(
                spec,
                base_policy=metadata,
                output_dir=output,
                python="python",
            )
            train_command = build_train_command(
                spec,
                base_policy=metadata,
                output_dir=output,
                python="python",
            )

            self.assertEqual(spec.rashomon_iters, 318_490)
            budget_index = safe_command.index("--rashomon-n-iters") + 1
            self.assertEqual(safe_command[budget_index], "318490")
            self.assertEqual(safe_command[safe_command.index("--seed") + 1], "7")
            self.assertEqual(train_command[train_command.index("--seed") + 1], "7")
            self.assertIn(str(base_path), safe_command)


if __name__ == "__main__":
    unittest.main()
