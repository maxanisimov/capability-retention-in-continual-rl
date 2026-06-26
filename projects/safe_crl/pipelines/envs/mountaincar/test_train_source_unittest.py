"""Unit tests for Mountain Car source training wiring."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from projects.safe_crl.pipelines.envs.mountaincar.core.methods import source_train


class MountainCarSourceTrainTests(unittest.TestCase):
    def test_build_actor_critic_uses_relu_and_expected_shapes(self) -> None:
        actor, critic = source_train.build_actor_critic(
            obs_dim=3,
            n_actions=3,
            hidden_size=64,
        )

        self.assertIsInstance(actor[1], torch.nn.ReLU)
        self.assertIsInstance(actor[3], torch.nn.ReLU)
        self.assertIsInstance(critic[1], torch.nn.ReLU)
        self.assertIsInstance(critic[3], torch.nn.ReLU)
        self.assertEqual(actor[0].in_features, 3)
        self.assertEqual(actor[-1].out_features, 3)
        self.assertEqual(critic[0].in_features, 3)
        self.assertEqual(critic[-1].out_features, 1)

    def test_arg_parser_defaults_match_requested_ppo_settings(self) -> None:
        args = source_train.build_arg_parser().parse_args([])

        self.assertEqual(args.total_timesteps, 200_000)
        self.assertEqual(args.rollout_steps, 2048)
        self.assertEqual(args.update_epochs, 10)
        self.assertEqual(args.minibatch_size, 64)
        self.assertAlmostEqual(args.gamma, 0.99)
        self.assertAlmostEqual(args.gae_lambda, 0.95)
        self.assertAlmostEqual(args.clip_coef, 0.2)
        self.assertAlmostEqual(args.ent_coef, 0.0)
        self.assertAlmostEqual(args.vf_coef, 0.5)
        self.assertAlmostEqual(args.lr, 3e-4)
        self.assertAlmostEqual(args.max_grad_norm, 0.5)
        self.assertEqual(args.hidden_size, 64)
        self.assertTrue(args.append_task_id)
        self.assertAlmostEqual(args.task_id, 0.0)
        self.assertEqual(args.eval_episodes_post_training, 100)
        self.assertAlmostEqual(args.solved_reward_threshold, -110.0)

    def test_main_writes_artifacts_and_summary_with_patched_training(self) -> None:
        captured: dict[str, object] = {}

        def fake_ppo_train(**kwargs):
            env = kwargs["env"]
            cfg = kwargs["cfg"]
            captured["cfg"] = cfg
            captured["train_obs_shape"] = env.observation_space.shape
            self.assertTrue(kwargs["return_training_data"])
            return (
                kwargs["actor_warm_start"],
                kwargs["critic_warm_start"],
                {
                    "states": np.zeros((1, 3), dtype=np.float32),
                    "actions": np.zeros((1,), dtype=np.int64),
                    "terminated": np.zeros((1,), dtype=np.float32),
                    "truncated": np.zeros((1,), dtype=np.float32),
                    "safe": np.ones((1,), dtype=np.float32),
                },
            )

        def fake_evaluate_with_success(*, env, actor, episodes, **kwargs):
            del actor, kwargs
            captured["eval_obs_shape"] = env.observation_space.shape
            captured["eval_episodes"] = episodes
            return -100.0, 4.0, 0.0, 1.0

        def fake_plot_episode(*, env, actor, save_path, **kwargs):
            del actor, kwargs
            captured["plot_obs_shape"] = env.observation_space.shape
            Path(save_path).write_bytes(b"plot")
            return []

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(source_train, "ppo_train", side_effect=fake_ppo_train),
                patch.object(
                    source_train,
                    "evaluate_with_success",
                    side_effect=fake_evaluate_with_success,
                ),
                patch.object(source_train, "plot_episode", side_effect=fake_plot_episode),
            ):
                source_train.main(["--output-dir", tmp_dir, "--seed", "7"])

            run_dir = Path(tmp_dir) / "default" / "seed_7" / "noadapt"
            actor_path = run_dir / "actor.pt"
            critic_path = run_dir / "critic.pt"
            training_data_path = run_dir / "training_data.pt"
            summary_path = run_dir / "run_summary.yaml"
            trajectory_path = run_dir / "trajectory_source.png"

            self.assertTrue(actor_path.is_file())
            self.assertTrue(critic_path.is_file())
            self.assertTrue(training_data_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(trajectory_path.is_file())

            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["activation"], "relu")
            self.assertEqual(summary["obs_dim"], 3)
            self.assertEqual(summary["n_actions"], 3)
            self.assertTrue(summary["append_task_id"])
            self.assertAlmostEqual(summary["task_id"], 0.0)
            self.assertEqual(summary["total_timesteps"], 200_000)
            self.assertEqual(summary["minibatch_size"], 64)
            self.assertAlmostEqual(summary["mean_reward"], -100.0)
            self.assertTrue(summary["solved"])
            self.assertEqual(summary["actor_path"], str(actor_path))
            self.assertEqual(summary["critic_path"], str(critic_path))
            self.assertEqual(summary["training_data_path"], str(training_data_path))
            self.assertEqual(summary["trajectory_source_plot_path"], str(trajectory_path))

            self.assertEqual(captured["train_obs_shape"], (3,))
            self.assertEqual(captured["eval_obs_shape"], (3,))
            self.assertEqual(captured["plot_obs_shape"], (3,))
            self.assertEqual(captured["eval_episodes"], 100)
            self.assertEqual(captured["cfg"].total_timesteps, 200_000)


if __name__ == "__main__":
    unittest.main()

