"""Historical precomputed/adaptive hyperparameter sweep tests."""

import tempfile
import unittest
from pathlib import Path
from projects.safe_policy_optimisation.archive.scripts import run_pspo_hparam_sweep

class HparamSweepTests(unittest.TestCase):
    def test_pspo_hparam_sweep_parser_accepts_two_swept_dimensions(self) -> None:
        args = run_pspo_hparam_sweep.parse_args(
            [
                "--env",
                "mini_pacman",
                "--method",
                "both",
                "--seeds",
                "0",
                "1",
                "--rashomon-iters",
                "100",
                "2000",
                "--bc-target-margins",
                "0.5",
                "1",
                "--n-hidden",
                "0",
                "--state-representation",
                "one_hot",
                "--bc-margin-mode",
                "all",
                "--rashomon-surrogate",
                "logsumexp",
                "--safe-region-shape",
                "zonotope",
                "--zonotope-rank",
                "4",
                "--dry-run",
            ]
        )

        self.assertEqual(args.env, "mini_pacman")
        self.assertEqual(args.method, "both")
        self.assertEqual(args.seeds, [0, 1])
        self.assertEqual(args.rashomon_iters, [100, 2000])
        self.assertEqual(args.bc_target_margins, [0.5, 1.0])
        self.assertEqual(args.n_hidden, 0)
        self.assertEqual(args.state_representation, "one_hot")
        self.assertEqual(args.bc_margin_mode, "all")
        self.assertEqual(args.rashomon_surrogate, "logsumexp")
        self.assertEqual(args.safe_region_shape, "zonotope")
        self.assertEqual(args.zonotope_rank, 4)


    def test_pspo_hparam_sweep_builds_only_method_iter_margin_grid(self) -> None:
        settings = run_pspo_hparam_sweep.build_settings(
            ["precomputed", "adaptive"],
            [100, 200],
            [0.5, 1.0],
        )

        self.assertEqual(len(settings), 8)
        self.assertEqual(settings[0].tag, "precomputed/iters_100__margin_0p5")
        self.assertEqual(settings[-1].tag, "adaptive/iters_200__margin_1")

        all_mode_settings = run_pspo_hparam_sweep.build_settings(
            ["precomputed"],
            [100],
            [0.5],
            bc_margin_mode="all",
            rashomon_surrogate="logsumexp",
        )

        self.assertEqual(
            all_mode_settings[0].tag,
            "precomputed/iters_100__margin_0p5__bc_all__surrogate_logsumexp",
        )


    def test_pspo_hparam_sweep_resolves_disjoint_cpu_slots(self) -> None:
        args = argparse.Namespace(
            cpu_ids=[2, 4, 6, 8],
            cores_per_setting=1,
            max_parallel=3,
        )
        slots = run_pspo_hparam_sweep.resolve_slots(args, n_settings=5)

        self.assertEqual(slots, [[2], [4], [6]])
        self.assertEqual(len({cpu for slot in slots for cpu in slot}), 3)


    def test_pspo_hparam_sweep_command_interprets_iters_by_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "sweep_root": tmp,
                "config": {
                    "shield_path": "projects/safe_policy_optimisation/artifacts/shield_q.pt",
                    "env_id": "CustomMiniPacman-v0",
                    "env_kwargs": {"ghost_rand_prob": 0.6},
                    "max_episode_steps": 1000,
                    "cost_limit": 0.01,
                    "total_timesteps": 500000,
                    "eval_episodes": 100,
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "early_stop_eval_freq": 0,
                    "early_stop_eval_episodes": 100,
                    "early_stop_success_rate": 1.0,
                    "success_reward_threshold": 0.0,
                    "curve_eval_freq": 2048,
                    "curve_eval_episodes": 20,
                    "rashomon_evaluation_policy": "unshielded",
                    "rashomon_checkpoint": 100,
                    "certificate_samples": 1000,
                },
                "state_representation": "one_hot",
                "device": "cpu",
                "hidden_dim": 64,
                "n_hidden": 0,
                "safety_demo_size": 8880,
                "adaptive_base_set_iters": 2000,
                "safe_region_shape": "zonotope",
                "zonotope_rank": 3,
            }
            pre = run_pspo_hparam_sweep.Setting(
                method="precomputed",
                rashomon_iters=10000,
                bc_target_margin=1.0,
            )
            adaptive = run_pspo_hparam_sweep.Setting(
                method="adaptive",
                rashomon_iters=100,
                bc_target_margin=1.0,
            )
            pre_all = run_pspo_hparam_sweep.Setting(
                method="precomputed",
                rashomon_iters=10000,
                bc_target_margin=1.0,
                bc_margin_mode="all",
            )

            pre_set_cmd = run_pspo_hparam_sweep.build_set_command(
                payload,
                setting=pre,
                n_iters=pre.rashomon_iters,
                cpu_ids=[2],
            )
            adaptive_set_cmd = run_pspo_hparam_sweep.build_set_command(
                payload,
                setting=adaptive,
                n_iters=payload["adaptive_base_set_iters"],
                cpu_ids=[4],
            )
            pre_train_cmd = run_pspo_hparam_sweep.build_precomputed_train_command(
                payload,
                setting=pre,
                seed=0,
                cpu_ids=[2],
            )
            adaptive_train_cmd = run_pspo_hparam_sweep.build_adaptive_train_command(
                payload,
                setting=adaptive,
                seed=0,
                cpu_ids=[4],
            )
            pre_all_set_cmd = run_pspo_hparam_sweep.build_set_command(
                payload,
                setting=pre_all,
                n_iters=pre_all.rashomon_iters,
                cpu_ids=[2],
            )

        self.assertEqual(
            pre_set_cmd[pre_set_cmd.index("--rashomon-n-iters") + 1],
            "10000",
        )
        self.assertEqual(
            adaptive_set_cmd[adaptive_set_cmd.index("--rashomon-n-iters") + 1],
            "2000",
        )
        self.assertEqual(
            adaptive_train_cmd[adaptive_train_cmd.index("--rashomon-n-iters") + 1],
            "100",
        )
        self.assertEqual(
            adaptive_train_cmd[adaptive_train_cmd.index("--rashomon-surrogate") + 1],
            "auto",
        )
        self.assertEqual(
            pre_set_cmd[pre_set_cmd.index("--bc-target-margin") + 1],
            "1.0",
        )
        self.assertEqual(
            pre_all_set_cmd[pre_all_set_cmd.index("--bc-margin-mode") + 1],
            "all",
        )
        self.assertIn("__bc_all", pre_all_set_cmd[pre_all_set_cmd.index("--output-dir") + 1])
        self.assertEqual(
            pre_set_cmd[pre_set_cmd.index("--rashomon-batch-size") + 1],
            "8880",
        )
        self.assertEqual(
            pre_set_cmd[pre_set_cmd.index("--safe-region-shape") + 1],
            "zonotope",
        )
        self.assertEqual(
            pre_set_cmd[pre_set_cmd.index("--zonotope-rank") + 1],
            "3",
        )
        self.assertEqual(
            pre_train_cmd[pre_train_cmd.index("--safe-region-shape") + 1],
            "zonotope",
        )
        self.assertEqual(
            adaptive_train_cmd[adaptive_train_cmd.index("--safe-region-shape") + 1],
            "zonotope",
        )
        self.assertEqual(
            pre_train_cmd[pre_train_cmd.index("--state-representation") + 1],
            "one_hot",
        )
        self.assertEqual(
            adaptive_train_cmd[adaptive_train_cmd.index("--state-representation") + 1],
            "one_hot",
        )
