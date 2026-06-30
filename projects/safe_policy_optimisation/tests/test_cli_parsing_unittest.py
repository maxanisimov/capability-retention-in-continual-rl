"""Tests for stage CLI parsers and pipeline settings composition."""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from projects.safe_policy_optimisation.run_experiment import (
    _synthesise_shield_if_needed,
    available_pipelines,
)
from projects.safe_policy_optimisation.run_experiment import (
    build_parser as build_experiment_launcher_parser,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    build_parser as build_shield_rashomon_parser,
)
from projects.safe_policy_optimisation.stages.rollout_policy_gif import (
    _is_masa_shielded_run_dir,
    _is_shielded_run_dir,
    _normalise_run_dir,
    _resolve_baseline_checkpoints,
)
from projects.safe_policy_optimisation.stages.rollout_policy_gif import (
    build_parser as build_rollout_parser,
)
from projects.safe_policy_optimisation.stages.synthesise_shield import (
    _resolve_max_episode_steps,
    _resolve_risk_threshold,
)
from projects.safe_policy_optimisation.stages.synthesise_shield import (
    build_parser as build_synthesise_shield_parser,
)
from projects.safe_policy_optimisation.stages.synthesise_shield import (
    default_output_dir as default_shield_output_dir,
)
from projects.safe_policy_optimisation.stages.train_policy_optimisation_pipeline import (
    apply_training_settings,
)
from projects.safe_policy_optimisation.stages.train_policy_optimisation_pipeline import (
    _baseline_worker_count,
    _pipeline_cpu_allocation,
    _policy_optimisation_method_count,
)
from projects.safe_policy_optimisation.stages.train_policy_optimisation_pipeline import (
    build_parser as build_deterministic_pipeline_parser,
)
from projects.safe_policy_optimisation.stages.train_masa_shielded_policy import (
    build_parser as build_masa_shielded_parser,
)
from projects.safe_policy_optimisation.stages.train_ppo import (
    build_parser as build_ppo_parser,
)
from projects.safe_policy_optimisation.stages.train_rashomon_shielded_policy import (
    build_parser as build_rashomon_shielded_ppo_parser,
)
from projects.safe_policy_optimisation.stages.train_cpo import (
    build_parser as build_cpo_parser,
)
from projects.safe_policy_optimisation.stages.train_ppo_lagrangian import (
    build_parser as build_train_parser,
)
from projects.safe_policy_optimisation.stages.train_discrete_shielded_policy import (
    build_parser as build_generic_shielded_parser,
)
from projects.safe_policy_optimisation.utils.config import (
    apply_settings_to_namespace,
    compose_pipeline_settings,
    load_task_registry,
)
from projects.safe_policy_optimisation.utils.cpu_allocation import (
    parse_cpu_ids,
    resolve_worker_count,
    worker_thread_count,
)
from projects.safe_policy_optimisation.utils.safe_rl import (
    EpisodeMetrics,
    aggregate_training_violations,
    aggregate_violations,
    build_safe_rl_baseline,
    evaluate_policy,
    make_minipacman_cost_fn,
    make_minipacman_env,
    make_safe_rl_env,
    minipacman_state_cost,
    save_gif,
    training_episode_rows,
)
from projects.safe_policy_optimisation.tests.helpers import (
    TwoStateEnv,
)

class CliParsingTests(unittest.TestCase):
    def test_train_parser_accepts_ppo_lagrangian_subset_and_cost_limit(self) -> None:
        args = build_train_parser().parse_args(
            [
                "--algorithms",
                "ppo_lagrangian",
                "--cost-limit",
                "1.0",
                "--eval-episodes",
                "7",
            ]
        )

        self.assertEqual(args.algorithms, ["ppo_lagrangian"])
        self.assertEqual(args.cost_limit, 1.0)
        self.assertEqual(args.eval_episodes, 7)
        self.assertEqual(args.early_stop_eval_freq, 0)

    def test_cpo_parser_is_cpo_only(self) -> None:
        args = build_cpo_parser().parse_args(["--algorithms", "cpo"])

        self.assertEqual(args.algorithms, ["cpo"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_cpo_parser().parse_args(["--algorithms", "ppo_lagrangian"])

    def test_train_parser_accepts_generic_env_settings(self) -> None:
        args = build_train_parser().parse_args(
            [
                "--env-id",
                "CustomMediaStreaming-v0",
                "--env-kwargs",
                '{"fast_rate": 0.0, "slow_rate": 0.0, "out_rate": 0.0}',
                "--max-episode-steps",
                "25",
            ]
        )

        self.assertEqual(args.env_id, "CustomMediaStreaming-v0")
        self.assertEqual(args.env_kwargs, {"fast_rate": 0.0, "slow_rate": 0.0, "out_rate": 0.0})
        self.assertEqual(args.max_episode_steps, 25)

    def test_train_parser_accepts_parallel_runtime_settings(self) -> None:
        args = build_train_parser().parse_args(
            [
                "--jobs",
                "3",
                "--torch-num-threads",
                "2",
                "--cpu-ids",
                "0,2,4",
                "--log-dir",
                "logs/baselines",
            ]
        )

        self.assertEqual(args.jobs, 3)
        self.assertEqual(args.torch_num_threads, 2)
        self.assertEqual(args.cpu_ids, [0, 2, 4])
        self.assertEqual(args.log_dir, Path("logs/baselines"))

    def test_old_environment_specific_stage_module_names_are_removed(self) -> None:
        old_modules = [
            "projects.safe_policy_optimisation.stages.train_deterministic_minipacman_pipeline",
            "projects.safe_policy_optimisation.stages.train_minipacman_safe_rl",
            "projects.safe_policy_optimisation.stages.train_minipacman_masa_shielded",
            "projects.safe_policy_optimisation.stages.train_minipacman_rashomon_shielded_ppo",
            "projects.safe_policy_optimisation.stages.train_shielded_policy",
            "projects.safe_policy_optimisation.stages.rollout_minipacman_policy_gif",
        ]
        for module_name in old_modules:
            with self.subTest(module_name=module_name), self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module_name)

    def test_train_parser_defaults_to_auto_parallelism(self) -> None:
        args = build_train_parser().parse_args([])

        self.assertEqual(args.jobs, 0)
        self.assertIsNone(args.cpu_ids)

    def test_train_parser_accepts_baseline_monitoring_settings(self) -> None:
        args = build_train_parser().parse_args(
            [
                "--shield-path",
                "shield_q.pt",
                "--tensorboard-log-dir",
                "tb/baselines",
                "--curve-eval-freq",
                "17",
                "--curve-eval-episodes",
                "6",
            ]
        )

        self.assertEqual(args.shield_path, Path("shield_q.pt"))
        self.assertEqual(args.tensorboard_log_dir, Path("tb/baselines"))
        self.assertEqual(args.curve_eval_freq, 17)
        self.assertEqual(args.curve_eval_episodes, 6)

    def test_deterministic_pipeline_parser_accepts_parallel_runtime_settings(self) -> None:
        args = build_deterministic_pipeline_parser().parse_args(
            [
                "--jobs",
                "4",
                "--torch-num-threads",
                "1",
                "--cpu-ids",
                "1,3",
                "--device",
                "auto",
            ]
        )

        self.assertEqual(args.jobs, 4)
        self.assertEqual(args.torch_num_threads, 1)
        self.assertEqual(args.cpu_ids, [1, 3])
        self.assertEqual(args.device, "auto")

    def test_deterministic_pipeline_parser_defaults_to_auto_parallelism(self) -> None:
        args = build_deterministic_pipeline_parser().parse_args([])

        self.assertEqual(args.jobs, 0)
        self.assertIsNone(args.cpu_ids)

    def test_auto_parallelism_counts_policy_methods(self) -> None:
        args = build_deterministic_pipeline_parser().parse_args([])
        method_count = _policy_optimisation_method_count(args)
        cpu_ids = list(range(method_count + 2))
        jobs = resolve_worker_count(args.jobs, method_count=method_count, cpu_ids=cpu_ids)
        baseline_jobs = _baseline_worker_count(args, jobs=jobs)
        allocation = _pipeline_cpu_allocation(
            args,
            jobs=jobs,
            baseline_jobs=baseline_jobs,
            cpu_ids=cpu_ids,
        )

        self.assertEqual(method_count, 6)
        self.assertEqual(jobs, method_count)
        self.assertEqual(baseline_jobs, 2)
        self.assertEqual(allocation["worker_cpu_ids"], cpu_ids[:method_count])
        self.assertEqual(allocation["strategy"], "dynamic_no_overlap")

    def test_parallelism_is_capped_by_available_cpu_ids(self) -> None:
        self.assertEqual(resolve_worker_count(0, method_count=5, cpu_ids=[0, 1]), 2)
        self.assertEqual(resolve_worker_count(3, method_count=5, cpu_ids=[0, 1]), 2)
        self.assertEqual(worker_thread_count(4, None), 1)
        self.assertEqual(worker_thread_count(1, None), None)
        self.assertEqual(parse_cpu_ids("3,5,3"), [3, 5])

    def test_deterministic_pipeline_parser_accepts_monitoring_settings(self) -> None:
        args = build_deterministic_pipeline_parser().parse_args(
            [
                "--tensorboard-log-dir",
                "tb",
                "--curve-eval-freq",
                "9",
                "--curve-eval-episodes",
                "4",
            ]
        )

        self.assertEqual(args.tensorboard_log_dir, Path("tb"))
        self.assertEqual(args.curve_eval_freq, 9)
        self.assertEqual(args.curve_eval_episodes, 4)

    def test_rollout_parser_accepts_checkpoint_and_episodes(self) -> None:
        args = build_rollout_parser().parse_args(
            [
                "--checkpoint",
                "run/ppo_lagrangian.pt",
                "--episodes",
                "4",
                "--fps",
                "8",
            ]
        )

        self.assertEqual(args.checkpoint, Path("run/ppo_lagrangian.pt"))
        self.assertEqual(args.episodes, 4)
        self.assertEqual(args.fps, 8.0)

    def test_rollout_detects_shielded_policy_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config.json").write_text('{"algorithm": "shielded_ppo"}\n', encoding="utf-8")
            (run_dir / "model.zip").write_bytes(b"placeholder")

            self.assertTrue(_is_shielded_run_dir(run_dir))

    def test_rollout_detects_rashomon_shielded_policy_model_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            model_path = run_dir / "model.zip"
            (run_dir / "config.json").write_text('{"algorithm": "rashomon_shielded_ppo"}\n', encoding="utf-8")
            model_path.write_bytes(b"placeholder")

            self.assertEqual(_normalise_run_dir(model_path), run_dir)
            self.assertTrue(_is_shielded_run_dir(model_path))

    def test_rollout_resolves_all_baseline_checkpoints_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            for name in ("cpo", "ppo_lagrangian", "ppo_pid_lagrangian"):
                (run_dir / f"{name}.pt").write_bytes(b"placeholder")
            args = build_rollout_parser().parse_args(["--run-dir", str(run_dir)])

            paths = _resolve_baseline_checkpoints(args)

            self.assertEqual(
                [path.name for path in paths],
                ["ppo_lagrangian.pt", "ppo_pid_lagrangian.pt", "cpo.pt"],
            )

    def test_rollout_accepts_direct_baseline_checkpoint_as_run_dir(self) -> None:
        args = build_rollout_parser().parse_args(["--run-dir", "cpo.pt"])

        self.assertEqual(_resolve_baseline_checkpoints(args), [Path("cpo.pt")])

    def test_rollout_detects_masa_shielded_policy_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "config.json").write_text('{"algorithm": "masa_shielded_ppo"}\n', encoding="utf-8")
            (run_dir / "model.zip").write_bytes(b"placeholder")

            self.assertTrue(_is_masa_shielded_run_dir(run_dir))

    def test_masa_shielded_parser_defaults_to_zero_tolerance(self) -> None:
        args = build_masa_shielded_parser().parse_args([])

        self.assertIsNone(args.env_id)
        self.assertEqual(args.safety_tolerance, 0.0)
        self.assertEqual(args.cost_limit, 0.0)

    def test_masa_shielded_parser_accepts_generic_env_settings(self) -> None:
        args = build_masa_shielded_parser().parse_args(
            [
                "--env-id",
                "CustomMediaStreaming-v0",
                "--env-kwargs",
                '{"fast_rate": 0.0, "slow_rate": 0.0, "out_rate": 0.0}',
            ]
        )

        self.assertEqual(args.env_id, "CustomMediaStreaming-v0")
        self.assertEqual(args.env_kwargs, '{"fast_rate": 0.0, "slow_rate": 0.0, "out_rate": 0.0}')

    def test_generic_shielded_parser_accepts_shield_and_env(self) -> None:
        args = build_generic_shielded_parser().parse_args(
            [
                "--shield-path",
                "shield_q.pt",
                "--env-id",
                "CustomMiniPacman-v0",
                "--env-kwargs",
                '{"ghost_rand_prob": 0.0}',
            ]
        )

        self.assertEqual(args.shield_path, Path("shield_q.pt"))
        self.assertEqual(args.env_id, "CustomMiniPacman-v0")
        self.assertEqual(args.env_kwargs, '{"ghost_rand_prob": 0.0}')
        self.assertEqual(args.shield_action_storage, "proposed")

    def test_plain_ppo_parser_accepts_env_without_shield(self) -> None:
        args = build_ppo_parser().parse_args(
            [
                "--env-id",
                "CustomMiniPacman-v0",
                "--env-kwargs",
                '{"ghost_rand_prob": 0.0}',
            ]
        )

        self.assertEqual(args.env_id, "CustomMiniPacman-v0")
        self.assertEqual(args.env_kwargs, '{"ghost_rand_prob": 0.0}')
        self.assertIsNone(args.shield_path)
        self.assertFalse(hasattr(args, "shield_action_storage"))

    def test_shielded_policy_parsers_accept_monitoring_settings(self) -> None:
        generic_args = build_generic_shielded_parser().parse_args(
            [
                "--shield-path",
                "shield_q.pt",
                "--env-id",
                "CustomMiniPacman-v0",
                "--tensorboard-log-dir",
                "tb/generic",
                "--curve-eval-freq",
                "11",
                "--curve-eval-episodes",
                "5",
            ]
        )
        rashomon_args = build_rashomon_shielded_ppo_parser().parse_args(
            [
                "--rashomon-dir",
                "rashomon_run",
                "--shield-path",
                "shield_q.pt",
                "--tensorboard-log-dir",
                "tb/rashomon",
                "--curve-eval-freq",
                "13",
                "--curve-eval-episodes",
                "6",
            ]
        )

        self.assertEqual(generic_args.tensorboard_log_dir, Path("tb/generic"))
        self.assertEqual(generic_args.curve_eval_freq, 11)
        self.assertEqual(generic_args.curve_eval_episodes, 5)
        self.assertEqual(rashomon_args.tensorboard_log_dir, Path("tb/rashomon"))
        self.assertEqual(rashomon_args.curve_eval_freq, 13)
        self.assertEqual(rashomon_args.curve_eval_episodes, 6)

    def test_synthesise_shield_parser_requires_explicit_env_and_task_settings(self) -> None:
        args = build_synthesise_shield_parser().parse_args([])

        self.assertIsNone(args.env)
        self.assertIsNone(args.task)
        self.assertFalse(hasattr(args, "risk_threshold"))
        self.assertEqual(args.constraint, "PCTL")

    def test_synthesise_shield_uses_alpha_as_risk_threshold(self) -> None:
        self.assertEqual(_resolve_risk_threshold({"alpha": 0.01}), 0.01)
        self.assertEqual(_resolve_risk_threshold({}), 0.0)

    def test_synthesise_shield_uses_task_max_episode_steps(self) -> None:
        self.assertEqual(
            _resolve_max_episode_steps("CustomMiniPacman-v0", "minipacman_default", None),
            100,
        )
        self.assertEqual(
            _resolve_max_episode_steps("CustomMiniPacman-v0", "minipacman_default", 7),
            7,
        )
        self.assertEqual(
            _resolve_max_episode_steps("CustomPacman-v0", "custompacman_paper_2503_07671", None),
            1000,
        )

    def test_synthesise_shield_rejects_risk_threshold_argument(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_synthesise_shield_parser().parse_args(["--risk-threshold", "0.0"])

    def test_synthesise_shield_accepts_masa_wrapper_arguments(self) -> None:
        args = build_synthesise_shield_parser().parse_args(
            [
                "--max-episode-steps",
                "100",
                "--init-safety-bound",
                "1e-12",
                "--theta",
                "1e-12",
                "--max-vi-steps",
                "2000",
                "--granularity",
                "10",
            ]
        )

        self.assertEqual(args.max_episode_steps, 100)
        self.assertEqual(args.init_safety_bound, 1e-12)
        self.assertEqual(args.theta, 1e-12)
        self.assertEqual(args.max_vi_steps, 2000)
        self.assertEqual(args.granularity, 10)

    def test_synthesise_shield_default_output_dir_is_project_local(self) -> None:
        path = default_shield_output_dir("CustomMiniPacman-v0", "minipacman_default")

        self.assertIn("safe_policy_optimisation", str(path))
        self.assertEqual(path.name, "minipacman_default")

    def test_shield_rashomon_parser_accepts_shield_path(self) -> None:
        args = build_shield_rashomon_parser().parse_args(
            [
                "--shield-path",
                "shield_q.pt",
                "--rashomon-n-iters",
                "0",
            ]
        )

        self.assertEqual(args.shield_path, Path("shield_q.pt"))
        self.assertEqual(args.rashomon_n_iters, 0)
        self.assertEqual(args.n_hidden, 0)

    def test_rashomon_shielded_ppo_parser_accepts_required_paths(self) -> None:
        args = build_rashomon_shielded_ppo_parser().parse_args(
            [
                "--rashomon-dir",
                "rashomon_run",
                "--shield-path",
                "shield_q.pt",
                "--env-id",
                "CustomMiniPacman-v0",
            ]
        )

        self.assertEqual(args.rashomon_dir, Path("rashomon_run"))
        self.assertEqual(args.shield_path, Path("shield_q.pt"))
        self.assertEqual(args.env_id, "CustomMiniPacman-v0")
        self.assertEqual(args.shield_action_storage, "proposed")
        self.assertEqual(args.early_stop_eval_policy, "unshielded")
        self.assertEqual(args.evaluation_policy, "unshielded")

    def test_training_parsers_accept_paper_hyperparameters(self) -> None:
        baseline_args = build_train_parser().parse_args(
            [
                "--gae-lambda",
                "0.9",
                "--clip-range",
                "0.1",
                "--max-grad-norm",
                "0.2",
                "--lagrangian-multiplier-init",
                "10.0",
            ]
        )
        shielded_args = build_generic_shielded_parser().parse_args(
            [
                "--shield-path",
                "shield_q.pt",
                "--env-id",
                "CustomMiniPacman-v0",
                "--gae-lambda",
                "0.9",
                "--clip-range",
                "0.1",
                "--max-grad-norm",
                "0.2",
            ]
        )
        rashomon_args = build_rashomon_shielded_ppo_parser().parse_args(
            [
                "--rashomon-dir",
                "rashomon_run",
                "--shield-path",
                "shield_q.pt",
                "--env-id",
                "CustomMiniPacman-v0",
                "--gae-lambda",
                "0.9",
                "--clip-range",
                "0.1",
                "--max-grad-norm",
                "0.2",
            ]
        )
        ppo_args = build_ppo_parser().parse_args(
            [
                "--env-id",
                "CustomMiniPacman-v0",
                "--gae-lambda",
                "0.9",
                "--clip-range",
                "0.1",
                "--max-grad-norm",
                "0.2",
            ]
        )

        self.assertEqual(baseline_args.gae_lambda, 0.9)
        self.assertEqual(baseline_args.clip_range, 0.1)
        self.assertEqual(baseline_args.max_grad_norm, 0.2)
        self.assertEqual(baseline_args.lagrangian_multiplier_init, 10.0)
        self.assertEqual(shielded_args.gae_lambda, 0.9)
        self.assertEqual(shielded_args.clip_range, 0.1)
        self.assertEqual(shielded_args.max_grad_norm, 0.2)
        self.assertEqual(rashomon_args.gae_lambda, 0.9)
        self.assertEqual(rashomon_args.clip_range, 0.1)
        self.assertEqual(rashomon_args.max_grad_norm, 0.2)
        self.assertEqual(ppo_args.gae_lambda, 0.9)
        self.assertEqual(ppo_args.clip_range, 0.1)
        self.assertEqual(ppo_args.max_grad_norm, 0.2)

    def test_build_safe_rl_baseline_applies_paper_hyperparameters(self) -> None:
        model = build_safe_rl_baseline(
            "ppo_lagrangian",
            TwoStateEnv(),
            cost_limit=0.05,
            seed=0,
            device="cpu",
            n_steps=16,
            batch_size=8,
            n_epochs=3,
            gae_lambda=0.9,
            clip_range=0.1,
            max_grad_norm=0.2,
            cost_gae_lambda=0.8,
            lagrangian_multiplier_init=10.0,
        )

        self.assertEqual(model.n_steps, 16)
        self.assertEqual(model.batch_size, 8)
        self.assertEqual(model.n_epochs, 3)
        self.assertEqual(model.gae_lambda, 0.9)
        self.assertEqual(model.clip_range, 0.1)
        self.assertEqual(model.max_grad_norm, 0.2)
        self.assertEqual(model.cost_gae_lambda, 0.8)
        self.assertEqual(model.lagrangian_multiplier, 10.0)

    def test_generic_shielded_ppo_defaults_to_unshielded_final_evaluation(self) -> None:
        args = build_generic_shielded_parser().parse_args(
            [
                "--shield-path",
                "shield_q.pt",
                "--env-id",
                "CustomMiniPacman-v0",
            ]
        )

        self.assertEqual(args.evaluation_policy, "unshielded")

    def test_deterministic_pipeline_defaults_to_deterministic_minipacman(self) -> None:
        args = build_deterministic_pipeline_parser().parse_args(["--pipeline", "deterministic_minipacman"])
        args = apply_training_settings(args)

        self.assertEqual(args.task, "minipacman_default")
        self.assertEqual(args.env_id, "CustomMiniPacman-v0")
        self.assertEqual(args.ghost_rand_prob, 0.0)
        self.assertEqual(args.max_episode_steps, 100)
        self.assertEqual(args.total_timesteps, 2_000)
        self.assertEqual(args.early_stop_success_rate, 1.0)
        self.assertEqual(args.shielded_evaluation_policy, "unshielded")
        self.assertEqual(args.rashomon_evaluation_policy, "unshielded")
        self.assertTrue(str(args.training_settings_file).endswith("deterministic/pipelines.yaml"))
        self.assertTrue(str(args.training_task_settings_file).endswith("deterministic/tasks.yaml"))

    def test_deterministic_pipeline_reads_rashomon_training_settings(self) -> None:
        args = build_deterministic_pipeline_parser().parse_args(
            [
                "--rashomon-dir",
                "projects/safe_policy_optimisation/artifacts/shield_rashomon/minipacman_default",
            ]
        )
        args = apply_training_settings(args, explicit_flags={"rashomon_dir"})

        self.assertEqual(args.run_id, "minipacman_default_yaml")
        self.assertEqual(args.total_timesteps, 2_000)
        self.assertEqual(args.algorithms, ["ppo_lagrangian", "ppo_pid_lagrangian", "cpo"])
        self.assertEqual(args.shielded_evaluation_policy, "unshielded")
        self.assertEqual(args.rashomon_evaluation_policy, "unshielded")
        self.assertTrue(str(args.training_settings_file).endswith("training_settings.yaml"))

    def test_deterministic_pipeline_cli_overrides_training_settings(self) -> None:
        args = build_deterministic_pipeline_parser().parse_args(
            [
                "--rashomon-dir",
                "projects/safe_policy_optimisation/artifacts/shield_rashomon/minipacman_default",
                "--total-timesteps",
                "7",
                "--run-id",
                "manual",
            ]
        )
        args = apply_training_settings(args, explicit_flags={"rashomon_dir", "total_timesteps", "run_id"})

        self.assertEqual(args.total_timesteps, 7)
        self.assertEqual(args.run_id, "manual")

    def test_experiment_launcher_lists_deterministic_minipacman(self) -> None:
        pipelines = available_pipelines()

        self.assertIn("deterministic_minipacman", pipelines)
        self.assertEqual(pipelines["deterministic_minipacman"].default_task, "minipacman_default")
        self.assertIn("train_policy_optimisation_pipeline", pipelines["deterministic_minipacman"].module)

    def test_environment_pipeline_registries_include_deterministic_pipelines(self) -> None:
        pipelines = available_pipelines()

        expected_pipelines = {
            "deterministic_bridge_crossing": "bridge_crossing_deterministic",
            "deterministic_bridge_crossing_v2": "bridge_crossing_v2_deterministic",
            "deterministic_colour_bomb": "colour_bomb_deterministic",
            "deterministic_colour_bomb_v2": "colour_bomb_v2_deterministic",
            "media_streaming_deterministic": "media_streaming_deterministic",
            "deterministic_minipacman": "minipacman_default",
        }
        for pipeline_name, task_name in expected_pipelines.items():
            self.assertEqual(pipelines[pipeline_name].default_task, task_name)

    def test_pipeline_settings_include_shield_synthesis_defaults(self) -> None:
        args = build_deterministic_pipeline_parser().parse_args(["--pipeline", "deterministic_minipacman"])
        args = apply_training_settings(args)

        self.assertEqual(args.constraint, "PCTL")
        self.assertEqual(args.constraint_kwargs, {"alpha": 0.01})
        self.assertEqual(args.max_vi_steps, 2000)
        self.assertEqual(args.granularity, 10)

    def test_environment_task_registries_include_deterministic_tasks(self) -> None:
        tasks = load_task_registry()

        expected_tasks = {
            "bridge_crossing_deterministic": "CustomBridgeCrossing-v0",
            "bridge_crossing_v2_deterministic": "CustomBridgeCrossingV2-v0",
            "colour_bomb_deterministic": "CustomColourBombGridWorld-v0",
            "colour_bomb_v2_deterministic": "CustomColourBombGridWorldV2-v0",
            "media_streaming_deterministic": "CustomMediaStreaming-v0",
            "minipacman_default": "CustomMiniPacman-v0",
        }
        for task_name, env_id in expected_tasks.items():
            self.assertEqual(tasks[task_name]["env_id"], env_id)

    def test_paper_pipeline_registries_include_main_experiment_pipelines(self) -> None:
        pipelines = available_pipelines()

        expected_pipelines = {
            "paper_2503_07671_media_streaming": "paper_2503_07671_media_streaming",
            "paper_2503_07671_colour_bomb": "paper_2503_07671_colour_bomb",
            "paper_2503_07671_colour_bomb_v2": "paper_2503_07671_colour_bomb_v2",
            "paper_2503_07671_bridge_crossing": "paper_2503_07671_bridge_crossing",
            "paper_2503_07671_bridge_crossing_v2": "paper_2503_07671_bridge_crossing_v2",
            "paper_2503_07671_pacman": "paper_2503_07671_pacman",
        }
        for pipeline_name, task_name in expected_pipelines.items():
            self.assertEqual(pipelines[pipeline_name].default_task, task_name)

    def test_paper_task_registries_include_main_experiment_tasks(self) -> None:
        tasks = load_task_registry()

        expected_tasks = {
            "paper_2503_07671_media_streaming": "CustomMediaStreamingV2-v0",
            "paper_2503_07671_colour_bomb": "CustomColourBombGridWorld-v0",
            "paper_2503_07671_colour_bomb_v2": "CustomColourBombGridWorldV3-v0",
            "paper_2503_07671_bridge_crossing": "CustomBridgeCrossing-v0",
            "paper_2503_07671_bridge_crossing_v2": "CustomBridgeCrossingV2-v0",
            "paper_2503_07671_pacman": "CustomPacman-v0",
        }
        for task_name, env_id in expected_tasks.items():
            self.assertEqual(tasks[task_name]["env_id"], env_id)

    def test_paper_pipeline_settings_apply_paper_hyperparameters(self) -> None:
        settings, _pipeline, task = compose_pipeline_settings("paper_2503_07671_colour_bomb")
        args = build_deterministic_pipeline_parser().parse_args([])
        args = apply_settings_to_namespace(args, settings, settings_file=Path("paper_2503_07671/pipelines.yaml"))

        self.assertEqual(task["shield_task"], "customcolourgridworld_paper_2503_07671")
        self.assertEqual(args.total_timesteps, 25_000)
        self.assertEqual(args.n_steps, 2048)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.n_epochs, 10)
        self.assertEqual(args.gae_lambda, 0.95)
        self.assertEqual(args.clip_range, 0.2)
        self.assertEqual(args.max_grad_norm, 0.5)
        self.assertEqual(args.cost_gae_lambda, 0.95)
        self.assertEqual(args.lagrangian_multiplier_init, 10.0)
        self.assertEqual(args.algorithms, ["ppo_lagrangian", "ppo_pid_lagrangian", "cpo"])
        self.assertEqual(args.early_stop_eval_freq, 2048)
        self.assertEqual(args.early_stop_eval_episodes, 100)
        self.assertEqual(args.early_stop_success_rate, 1.0)
        self.assertEqual(args.curve_eval_freq, 2048)
        self.assertEqual(args.curve_eval_episodes, 20)
        self.assertFalse(args.skip_rashomon_policy)
        self.assertEqual(args.shielded_evaluation_policy, "shielded")

    def test_all_paper_pipelines_enable_success_rate_early_stopping(self) -> None:
        for pipeline_name in available_pipelines():
            if not pipeline_name.startswith("paper_2503_07671_"):
                continue
            with self.subTest(pipeline_name=pipeline_name):
                settings, _pipeline, _task = compose_pipeline_settings(pipeline_name)
                args = build_deterministic_pipeline_parser().parse_args([])
                args = apply_settings_to_namespace(
                    args,
                    settings,
                    settings_file=Path("paper_2503_07671/pipelines.yaml"),
                )

                self.assertEqual(args.early_stop_eval_freq, 2048)
                self.assertEqual(args.early_stop_eval_episodes, 100)
                self.assertEqual(args.early_stop_success_rate, 1.0)

    def test_paper_env_settings_match_local_state_and_action_spaces(self) -> None:
        expected = {
            "paper_2503_07671_media_streaming": (462, 2),
            "paper_2503_07671_colour_bomb": (81, 5),
            "paper_2503_07671_colour_bomb_v2": (900, 5),
            "paper_2503_07671_bridge_crossing": (400, 5),
            "paper_2503_07671_bridge_crossing_v2": (400, 5),
        }
        for pipeline_name, (n_states, n_actions) in expected.items():
            settings, _pipeline, _task = compose_pipeline_settings(pipeline_name)
            env = make_safe_rl_env(
                settings["env_id"],
                max_episode_steps=settings["max_episode_steps"],
                env_kwargs=settings["env_kwargs"],
            )
            try:
                self.assertEqual(env.unwrapped.observation_space.n, n_states)
                self.assertEqual(env.unwrapped.action_space.n, n_actions)
            finally:
                env.close()

    def test_paper_pacman_uses_large_sparse_local_state_space(self) -> None:
        settings, _pipeline, _task = compose_pipeline_settings("paper_2503_07671_pacman")
        env = make_safe_rl_env(
            settings["env_id"],
            max_episode_steps=settings["max_episode_steps"],
            env_kwargs=settings["env_kwargs"],
        )
        try:
            self.assertGreater(env.unwrapped.observation_space.n, 100_000)
            self.assertEqual(env.unwrapped.action_space.n, 5)
        finally:
            env.close()

    def test_media_streaming_pipeline_uses_project_task_with_masa_shield_alias(self) -> None:
        settings, _pipeline, task = compose_pipeline_settings("media_streaming_deterministic")

        self.assertEqual(settings["task"], "media_streaming_deterministic")
        self.assertEqual(settings["env_id"], "CustomMediaStreaming-v0")
        self.assertEqual(task["shield_task"], "custommediastreaming_default")

    def test_experiment_launcher_passes_unknown_flags_to_pipeline(self) -> None:
        args, passthrough = build_experiment_launcher_parser().parse_known_args(
            [
                "--pipeline",
                "deterministic_minipacman",
                "--task",
                "minipacman_default",
                "--run-id",
                "manual",
                "--total-timesteps",
                "7",
            ]
        )

        self.assertEqual(args.pipeline, "deterministic_minipacman")
        self.assertEqual(args.task, "minipacman_default")
        self.assertEqual(args.run_id, "manual")
        self.assertEqual(passthrough, ["--total-timesteps", "7"])

    def test_experiment_launcher_reuses_existing_shield_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shield_path = Path(tmpdir) / "shield_q.pt"
            shield_path.write_bytes(b"placeholder")
            result = _synthesise_shield_if_needed(
                {
                    "shield_path": shield_path,
                    "env_id": "CustomMiniPacman-v0",
                    "task": "minipacman_default",
                },
                task_settings={"env_kwargs": {"ghost_rand_prob": 0.0}},
            )

            self.assertEqual(result, shield_path)

    def test_experiment_launcher_requires_valid_task_env_kwargs_for_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaises(ValueError):
            _synthesise_shield_if_needed(
                {
                    "shield_path": Path(tmpdir) / "shield_q.pt",
                    "env_id": "CustomMiniPacman-v0",
                    "task": "minipacman_default",
                },
                task_settings={"env_kwargs": []},
            )


if __name__ == "__main__":
    unittest.main()
