"""Tests for the MountainCar edge-safety certification stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.safe_policy_optimisation.stages.train_mountaincar_edge_safety import (
    LEFT_ACTION,
    RIGHT_ACTION,
    build_parser,
    make_edge_shield,
    make_safe_parameter_dataset,
    run_experiment,
)


class MountainCarEdgeSafetyTests(unittest.TestCase):
    def test_edge_shield_classifies_both_edges_and_fallback(self) -> None:
        shield = make_edge_shield(left_threshold=-1.05, right_threshold=0.45, seed=0)

        states = shield.obs_to_state(
            np.array(
                [
                    [-1.1, 0.0],
                    [0.5, 0.0],
                    [-0.4, 0.0],
                ],
                dtype=np.float32,
            )
        )

        self.assertEqual(states.tolist(), [0, 1, 2])
        self.assertEqual(shield.safe_actions(0).tolist(), [RIGHT_ACTION])
        self.assertEqual(shield.safe_actions(1).tolist(), [LEFT_ACTION])
        self.assertEqual(shield.mask[2].tolist(), [1, 1, 1])

    def test_parser_accepts_smoke_settings(self) -> None:
        args = build_parser().parse_args(
            [
                "--run-id",
                "smoke",
                "--total-timesteps",
                "0",
                "--eval-episodes",
                "1",
                "--net-arch",
                "--safe-init-samples",
                "128",
                "--learning-rate",
                "0.004",
                "--gamma",
                "0.98",
                "--train-freq",
                "16",
                "--gradient-steps",
                "8",
                "--target-update-interval",
                "600",
                "--exploration-fraction",
                "0.2",
                "--exploration-final-eps",
                "0.07",
                "--rashomon-n-iters",
                "0",
            ]
        )

        self.assertEqual(args.run_id, "smoke")
        self.assertEqual(args.total_timesteps, 0)
        self.assertEqual(args.eval_episodes, 1)
        self.assertEqual(args.net_arch, [])
        self.assertEqual(args.safe_init_samples, 128)
        self.assertEqual(args.learning_rate, 0.004)
        self.assertEqual(args.gamma, 0.98)
        self.assertEqual(args.train_freq, 16)
        self.assertEqual(args.gradient_steps, 8)
        self.assertEqual(args.target_update_interval, 600)
        self.assertEqual(args.exploration_fraction, 0.2)
        self.assertEqual(args.exploration_final_eps, 0.07)
        self.assertEqual(args.rashomon_n_iters, 0)

    def test_safe_parameter_dataset_uses_edge_boxes_and_masks(self) -> None:
        import gymnasium as gym
        from provably_safe_policy_optimisation import ProvablySafeDQN

        shield = make_edge_shield(left_threshold=-1.05, right_threshold=0.45, seed=0)
        model = ProvablySafeDQN(
            "MlpPolicy",
            gym.make("MountainCar-v0"),
            device="cpu",
            shield=shield,
            seed=0,
            shield_seed=0,
            policy_kwargs={"net_arch": []},
        )
        self.addCleanup(model.get_env().close)

        dataset, metadata = make_safe_parameter_dataset(model, shield)
        x_l, x_u, masks = dataset.tensors

        self.assertEqual(metadata["dataset_size"], 2)
        self.assertTrue(np.allclose(x_l[0].numpy(), [-1.2, -0.07]))
        self.assertTrue(np.allclose(x_u[0].numpy(), [-1.05, 0.07]))
        self.assertEqual(masks.int().tolist(), [[0, 0, 1], [1, 0, 0]])

    def test_run_experiment_writes_artifacts_and_certifies_initial_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "--run-id",
                    "smoke",
                    "--output-dir",
                    tmpdir,
                    "--total-timesteps",
                    "0",
                    "--eval-episodes",
                    "1",
                    "--net-arch",
                    "--safe-init-samples",
                    "256",
                    "--safe-init-bc-epochs",
                    "300",
                    "--safe-init-refine-epochs",
                    "500",
                    "--safe-init-lr",
                    "0.05",
                    "--rashomon-n-iters",
                    "0",
                    "--rashomon-checkpoint",
                    "1",
                    "--rashomon-batch-size",
                    "2",
                    "--certificate-samples",
                    "2",
                ]
            )

            summary = run_experiment(args)
            run_dir = Path(tmpdir) / "smoke"

            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "episodes.csv").exists())
            self.assertTrue((run_dir / "model.zip").exists())
            self.assertTrue((run_dir / "safe_parameter_dataset.pt").exists())
            self.assertTrue((run_dir / "safe_param_bounds.pt").exists())
            self.assertTrue((run_dir / "safe_parameter_bounded_model.pt").exists())
            self.assertTrue(summary["pretrain_certificate"]["all_certified"])
            self.assertEqual(summary["pretrain_certificate"]["certified_fraction"], 1.0)
            self.assertTrue(summary["safe_parameter_space"]["attached_to_optimizer"])
            self.assertEqual(summary["safe_parameter_space"]["selected_certificate"], 1.0)
            self.assertTrue(summary["projection"]["active"])
            self.assertTrue(summary["projection"]["is_within_bounds"])
            self.assertTrue(summary["final_certificate"]["all_certified"])

    def test_projected_training_keeps_policy_inside_safe_parameter_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = build_parser().parse_args(
                [
                    "--run-id",
                    "projected",
                    "--output-dir",
                    tmpdir,
                    "--total-timesteps",
                    "80",
                    "--learning-starts",
                    "1",
                    "--buffer-size",
                    "200",
                    "--batch-size",
                    "8",
                    "--learning-rate",
                    "0.004",
                    "--gamma",
                    "0.98",
                    "--train-freq",
                    "16",
                    "--gradient-steps",
                    "8",
                    "--target-update-interval",
                    "600",
                    "--exploration-fraction",
                    "0.2",
                    "--exploration-final-eps",
                    "0.07",
                    "--eval-episodes",
                    "0",
                    "--net-arch",
                    "--safe-init-samples",
                    "256",
                    "--safe-init-bc-epochs",
                    "300",
                    "--safe-init-refine-epochs",
                    "500",
                    "--safe-init-lr",
                    "0.05",
                    "--rashomon-n-iters",
                    "0",
                    "--rashomon-checkpoint",
                    "1",
                    "--rashomon-batch-size",
                    "2",
                    "--certificate-samples",
                    "2",
                ]
            )

            summary = run_experiment(args)

            self.assertTrue(summary["projection"]["active"])
            self.assertTrue(summary["projection"]["is_within_bounds"])
            self.assertEqual(summary["projection"]["max_violation"], 0.0)
            self.assertGreater(summary["projection"]["diagnostics"]["bounded_steps"], 0)
            self.assertEqual(summary["config"]["dqn"]["learning_rate"], 0.004)
            self.assertEqual(summary["config"]["dqn"]["gamma"], 0.98)
            self.assertEqual(summary["config"]["dqn"]["train_freq"], 16)
            self.assertEqual(summary["config"]["dqn"]["gradient_steps"], 8)
            self.assertEqual(summary["config"]["dqn"]["target_update_interval"], 600)
            self.assertEqual(summary["config"]["dqn"]["exploration_fraction"], 0.2)
            self.assertEqual(summary["config"]["dqn"]["exploration_final_eps"], 0.07)


if __name__ == "__main__":
    unittest.main()
