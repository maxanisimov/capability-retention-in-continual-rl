"""RL-SGF factory and CLI wiring."""

from __future__ import annotations

import argparse
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import torch

from projects.safe_policy_optimisation.stages import train_ppo_lagrangian, train_rl_sgf
from projects.safe_policy_optimisation.utils.safe_rl import (
    build_safe_rl_baseline, save_checkpoint, load_checkpoint_model,
    training_episode_rows, write_training_episode_csv,
)
from projects.safe_policy_optimisation.scripts.experiments import run_baselines_pspo_init_ablation as sweep
from safe_rl_baselines import CPO, RLSGF


class RLSGFWiringTests(unittest.TestCase):
    def test_training_csv_retains_both_sampling_phases(self):
        import csv
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episodes.csv"
            records = [dict(episode=i, end_timestep=i+1, reward=1.0, cost=0.0, length=1,
                            violated=False, sampling_phase=phase)
                       for i, phase in enumerate(("ppo", "constraint"))]
            write_training_episode_csv(path, training_episode_rows("rl_sgf", records))
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sampling_phase"] for row in rows], ["ppo", "constraint"])

    def test_factory_forwards_rl_sgf_hyperparameters(self) -> None:
        model = build_safe_rl_baseline(
            "rl_sgf", gym.make("CartPole-v1"), cost_limit=1.0, seed=0, net_arch=(16,),
            rl_sgf_step_size=0.2, rl_sgf_alpha=2.0, rl_sgf_episodes_per_iter=7, rl_sgf_baseline="critic",
            cost_gamma=0.95,
        )
        self.assertIsInstance(model, RLSGF)
        self.assertEqual((model.step_size, model.alpha, model.episodes_per_iter, model.baseline),
                         (0.2, 2.0, 7, "critic"))
        self.assertEqual(model.cost_gamma, 0.95)
        self.assertEqual(model.update_mode, "ppo")

    def test_other_baselines_ignore_rl_sgf_keys(self) -> None:
        model = build_safe_rl_baseline(
            "cpo", gym.make("CartPole-v1"), cost_limit=1.0, seed=0, net_arch=(16,), rl_sgf_step_size=0.2,
        )
        self.assertIsInstance(model, CPO)

    def test_stage_parser_and_hyperparameter_forwarding(self) -> None:
        args = train_rl_sgf.build_parser().parse_args(
            ["--env-id", "CartPole-v1", "--rl-sgf-step-size", "0.3", "--rl-sgf-episodes-per-iter", "11"]
        )
        hp = train_ppo_lagrangian._baseline_hyperparameters_from_args(args)
        self.assertEqual(hp["rl_sgf_step_size"], 0.3)
        self.assertEqual(hp["rl_sgf_episodes_per_iter"], 11)
        self.assertEqual(hp["rl_sgf_baseline"], "none")
        self.assertEqual(hp["rl_sgf_update_mode"], "ppo")
        self.assertEqual(args.algorithms, ["rl_sgf"])

    def test_ppo_lagrangian_parser_still_builds_hyperparameters(self) -> None:
        args = train_ppo_lagrangian.build_parser().parse_args(["--env-id", "CartPole-v1"])
        hp = train_ppo_lagrangian._baseline_hyperparameters_from_args(args)
        self.assertIsNone(hp["rl_sgf_step_size"])
        self.assertIsInstance(args, argparse.Namespace)

    def test_native_mode_can_be_selected_through_cli_and_factory(self):
        args = train_rl_sgf.build_parser().parse_args(
            ["--env-id", "CartPole-v1", "--rl-sgf-update-mode", "reinforce"]
        )
        model = build_safe_rl_baseline("rl_sgf", gym.make("CartPole-v1"), cost_limit=1.0,
                                       seed=0, **train_ppo_lagrangian._baseline_hyperparameters_from_args(args))
        self.assertEqual(model.update_mode, "reinforce")

    def test_checkpoint_round_trip_preserves_both_variants_and_hyperparameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ("ppo", "reinforce"):
                env = gym.make("CartPole-v1")
                self.addCleanup(env.close)
                model = build_safe_rl_baseline("rl_sgf", env, cost_limit=1.0, seed=0,
                                               net_arch=(16,), rl_sgf_update_mode=mode,
                                               rl_sgf_step_size=0.2, rl_sgf_episodes_per_iter=3,
                                               n_steps=8, n_epochs=2)
                model.ppo_timesteps = 8 if mode == "ppo" else 0
                path = Path(temporary) / f"{mode}.pt"
                save_checkpoint(path, model, algorithm="rl_sgf", metadata={"cost_limit": 1.0, "seed": 0})
                restored, payload = load_checkpoint_model(path, env=env)
                self.assertEqual(restored.update_mode, mode)
                self.assertEqual(restored.step_size, 0.2)
                self.assertEqual(restored.episodes_per_iter, 3)
                self.assertEqual(restored.n_steps, 8)
                self.assertEqual(restored.n_epochs, 2)
                self.assertEqual(restored.ppo_timesteps, model.ppo_timesteps)
                self.assertEqual(payload["metadata"]["rl_sgf_update_mode"], mode)
                torch.testing.assert_close(restored._flat_params(), model._flat_params(), rtol=0, atol=0)

    def test_legacy_checkpoint_loads_as_reinforce(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = gym.make("CartPole-v1")
            self.addCleanup(env.close)
            model = build_safe_rl_baseline("rl_sgf", env, cost_limit=1.0, seed=0)
            path = Path(temporary) / "legacy.pt"
            save_checkpoint(path, model, algorithm="rl_sgf", metadata={"cost_limit": 1.0})
            payload = torch.load(path, weights_only=False)
            payload["metadata"] = {"cost_limit": 1.0}
            torch.save(payload, path)
            restored, _ = load_checkpoint_model(path, env=env)
            self.assertEqual(restored.update_mode, "reinforce")

    def test_baseline_sweep_separates_modes_and_forwards_option(self):
        for mode, directory in (("ppo", "rl_sgf_ppo"), ("reinforce", "rl_sgf")):
            command, run_dir = sweep.build_command("media_streaming", "rl_sgf", 0,
                output_root=Path("/tmp/sgf-mode-test"), warm_start=False, smoke=True,
                rl_sgf_update_mode=mode)
            self.assertEqual(run_dir.parent.name, directory)
            self.assertEqual(command[command.index("--rl-sgf-update-mode") + 1], mode)

    def test_baseline_sweep_does_not_skip_results_from_wrong_mode(self):
        args = sweep.build_parser().parse_args(["--methods", "rl_sgf", "--envs", "media_streaming",
                                              "--seeds", "0", "--no-cold-start"])
        with tempfile.TemporaryDirectory() as temporary:
            args.output_root = Path(temporary)
            run_dir = args.output_root / "seed0"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text('{}')
            (run_dir / "config.json").write_text('{}')
            with patch.object(sweep, "build_command", return_value=(["python"], run_dir)):
                with self.assertRaisesRegex(ValueError, "mode differs"):
                    sweep.plan_jobs(args)


if __name__ == "__main__":
    unittest.main()
