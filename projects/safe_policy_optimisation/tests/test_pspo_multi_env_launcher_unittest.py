"""Tests for the multi-environment PSPO launcher."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from projects.safe_policy_optimisation.scripts.launch_pspo_multi_env import (
    DEFAULT_ENVS,
    build_launch_environment,
    build_parser,
    parse_launcher_args,
    parse_mpstat_idle,
    safety_demo_sizes,
    select_idle_cpus,
)
from projects.safe_policy_optimisation.utils.pspo_defaults import (
    environment_defaults,
)


class PspoMultiEnvLauncherTests(unittest.TestCase):
    def test_environment_defaults_match_recorded_best_runs(self) -> None:
        expected = {
            "media_streaming": (25_000, "1"),
            "colour_bomb": (25_000, "1"),
            "colour_bomb_v2": (100_000, "1"),
            "bridge_crossing": (200_000, "1"),
            "bridge_crossing_v2": (1_600_000, "1"),
            "mini_pacman": (2_000_000, "100"),
        }
        self.assertEqual(
            {
                environment: (
                    environment_defaults(environment).total_timesteps,
                    environment_defaults(environment).frequency,
                )
                for environment in DEFAULT_ENVS
            },
            expected,
        )

    def test_cli_help_groups_related_arguments(self) -> None:
        help_text = build_parser().format_help()

        headings = (
            "experiment selection:",
            "PPO update settings:",
            "policy initialisation:",
            "LID settings:",
            "CPU allocation and execution:",
        )
        positions = [help_text.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))

        sections = {
            heading: help_text[start:end]
            for (heading, start), end in zip(
                zip(headings, positions),
                positions[1:] + [len(help_text)],
            )
        }
        self.assertIn("--envs", sections["experiment selection:"])
        self.assertIn("--freq", sections["PPO update settings:"])
        self.assertNotIn("--adaptive-granularity", help_text)
        self.assertIn(
            "--bc-safe-action-entropy-weight",
            sections["policy initialisation:"],
        )
        self.assertIn(
            "--lid-objective",
            sections["LID settings:"],
        )
        self.assertIn("--minimum-idle", sections["CPU allocation and execution:"])

    def test_lid_cli_names_replace_rashomon_names_in_help(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()

        self.assertIn("--lid-n-iters", help_text)
        self.assertIn("--lid-objective", help_text)
        self.assertNotIn("--rashomon-n-iters", help_text)
        self.assertNotIn("--rashomon-objective", help_text)

        lid_args = parser.parse_args(
            ["--lid-n-iters", "321", "--lid-objective", "projection_distance"]
        )
        self.assertEqual(lid_args.rashomon_n_iters, 321)
        self.assertEqual(lid_args.rashomon_objective, "projection_distance")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            legacy_args = parse_launcher_args(
                [
                    "--rashomon-n-iters",
                    "123",
                    "--rashomon-objective",
                    "projection_distance",
                ]
            )
        self.assertEqual(legacy_args.rashomon_n_iters, 123)
        self.assertEqual(legacy_args.rashomon_objective, "projection_distance")
        self.assertIn(
            "--rashomon-n-iters is deprecated; use --lid-n-iters instead",
            stderr.getvalue(),
        )
        self.assertIn(
            "--rashomon-objective is deprecated; use --lid-objective instead",
            stderr.getvalue(),
        )

    def test_legacy_lid_aliases_cannot_be_mixed_with_canonical_names(self) -> None:
        conflicting_options = (
            ["--lid-n-iters", "200", "--rashomon-n-iters", "100"],
            [
                "--lid-objective",
                "weighted_width",
                "--rashomon-objective",
                "projection_distance",
            ],
        )
        for options in conflicting_options:
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_launcher_args(options)

    def test_architecture_cli_defaults_to_two_hidden_and_accepts_one_hidden(self) -> None:
        self.assertEqual(build_parser().parse_args([]).architecture, "two_hidden")
        self.assertEqual(
            build_parser().parse_args(["--architecture", "one_hidden"]).architecture,
            "one_hidden",
        )

    def test_full_safety_demonstration_sizes(self) -> None:
        self.assertEqual(
            safety_demo_sizes(list(DEFAULT_ENVS)),
            {
                "media_streaming": 441,
                "colour_bomb": 74,
                "colour_bomb_v2": 856,
                "bridge_crossing": 332,
                "bridge_crossing_v2": 343,
                "mini_pacman": 8880,
            },
        )

    def test_media_streaming_is_a_supported_default_environment(self) -> None:
        args = build_parser().parse_args(["--envs", "media_streaming"])

        self.assertIn("media_streaming", DEFAULT_ENVS)
        self.assertEqual(args.envs, ["media_streaming"])

    def test_mpstat_parser_and_idle_selection(self) -> None:
        output = """
Average:     CPU    %usr   %nice    %sys %iowait   %idle
Average:       0    1.00    0.00    1.00    0.00   98.00
Average:       1   20.00    0.00    1.00    0.00   79.00
Average:       2    0.00    0.00    0.00    0.00  100.00
"""
        idle = parse_mpstat_idle(output)

        self.assertEqual(idle, {0: 98.0, 1: 79.0, 2: 100.0})
        self.assertEqual(
            select_idle_cpus(
                idle,
                required=2,
                minimum_idle=90.0,
                allowed_cpus={0, 1, 2},
            ),
            [2, 0],
        )
        with self.assertRaises(RuntimeError):
            select_idle_cpus(
                idle,
                required=3,
                minimum_idle=90.0,
                allowed_cpus={0, 1, 2},
            )

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_environment_locks_requested_experiment_settings(self) -> None:
        env = build_launch_environment(
            environment="mini_pacman",
            seeds=[0, 1],
            cpu_ids=[17, 23],
            architecture="two_hidden",
            run_name="test_run",
            n_iters=200,
            dry_run=True,
        )

        self.assertEqual(env["CPU_IDS"], "17,23")
        self.assertEqual(env["SEEDS"], "0 1")
        self.assertEqual(env["ARCHITECTURE"], "two_hidden")
        self.assertEqual(env["REGION_MODE"], "replace")
        self.assertEqual(env["RASHOMON_MULTI_LABEL_MODE"], "all")
        self.assertEqual(env["RASHOMON_SURROGATE"], "logsumexp")
        self.assertEqual(env["RASHOMON_OBJECTIVE"], "weighted_width")
        self.assertEqual(env["VERIFY_FIRST"], "false")
        self.assertEqual(env["RASHOMON_BATCH_SIZE"], "auto")
        self.assertEqual(env["RASHOMON_CERTIFICATE_SAMPLES"], "all")
        self.assertEqual(env["RASHOMON_N_ITERS"], "200")
        self.assertEqual(env["BC_TARGET_MARGIN"], "2.0")
        self.assertEqual(env["BC_SAFE_ACTION_ENTROPY_WEIGHT"], "1.0")
        self.assertEqual(env["BC_MIN_SAFE_ACTION_ENTROPY"], "0.95")
        self.assertEqual(env["BC_INITIALISATION_OBJECTIVE"], "margin")
        self.assertEqual(env["BC_UNSAFE_MASS_TARGET"], "0.01")
        self.assertEqual(env["BC_MAX_UNSAFE_MASS"], "0.02")
        self.assertEqual(env["BC_SAFE_ACTION_UNIFORMITY_WEIGHT"], "1.0")
        self.assertEqual(env["DIRECTIONAL_RASHOMON_GROWTH"], "1")
        self.assertNotIn("ADAPTIVE_GRANULARITY", env)
        self.assertEqual(env["ADAPTIVE_FREQ"], "100")
        self.assertEqual(env["TOTAL_TIMESTEPS"], "2000000")
        self.assertEqual(env["STOP_WHEN_PROPOSAL_CONTAINED"], "1")
        self.assertEqual(env["DRY_RUN"], "1")

        one_hidden = build_launch_environment(
            environment="mini_pacman",
            seeds=[0],
            cpu_ids=[17],
            architecture="one_hidden",
            run_name="one_hidden_run",
            n_iters=200,
            dry_run=True,
        )
        self.assertEqual(one_hidden["ARCHITECTURE"], "one_hidden")

        projection = build_launch_environment(
            environment="mini_pacman",
            seeds=[0],
            cpu_ids=[17],
            architecture="two_hidden",
            run_name="projection_run",
            n_iters=200,
            dry_run=True,
            adaptive_freq="rollout",
            rashomon_objective="projection_distance",
        )
        self.assertEqual(projection["RASHOMON_OBJECTIVE"], "projection_distance")

        verify_first = build_launch_environment(
            environment="mini_pacman",
            seeds=[0],
            cpu_ids=[17],
            architecture="two_hidden",
            run_name="verify_first_run",
            n_iters=200,
            dry_run=True,
            verify_first=True,
        )
        self.assertEqual(verify_first["VERIFY_FIRST"], "true")

        entropy = build_launch_environment(
            environment="mini_pacman",
            seeds=[0],
            cpu_ids=[17],
            architecture="two_hidden",
            run_name="entropy_run",
            n_iters=200,
            dry_run=True,
            bc_safe_action_entropy_weight=1.0,
            bc_min_safe_action_entropy=0.97,
        )
        self.assertEqual(entropy["BC_SAFE_ACTION_ENTROPY_WEIGHT"], "1.0")
        self.assertEqual(entropy["BC_MIN_SAFE_ACTION_ENTROPY"], "0.97")

        safe_mass = build_launch_environment(
            environment="mini_pacman",
            seeds=[0],
            cpu_ids=[17],
            architecture="two_hidden",
            run_name="safe_mass_run",
            n_iters=200,
            dry_run=True,
            bc_initialisation_objective="safe_mass",
            bc_unsafe_mass_target=0.02,
            bc_max_unsafe_mass=0.03,
            bc_safe_action_uniformity_weight=1.5,
        )
        self.assertEqual(safe_mass["BC_INITIALISATION_OBJECTIVE"], "safe_mass")
        self.assertEqual(safe_mass["BC_UNSAFE_MASS_TARGET"], "0.02")
        self.assertEqual(safe_mass["BC_MAX_UNSAFE_MASS"], "0.03")
        self.assertEqual(safe_mass["BC_SAFE_ACTION_UNIFORMITY_WEIGHT"], "1.5")

    def test_safe_action_entropy_cli_is_enabled_by_default(self) -> None:
        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.bc_safe_action_entropy_weight, 1.0)
        self.assertEqual(defaults.bc_min_safe_action_entropy, 0.95)
        self.assertEqual(defaults.bc_initialisation_objective, "margin")
        self.assertEqual(defaults.bc_unsafe_mass_target, 0.01)
        self.assertEqual(defaults.bc_max_unsafe_mass, 0.02)
        self.assertEqual(defaults.bc_safe_action_uniformity_weight, 1.0)

        overridden = build_parser().parse_args(
            [
                "--bc-safe-action-entropy-weight",
                "0.0",
                "--bc-min-safe-action-entropy",
                "0.97",
                "--bc-initialisation-objective",
                "safe_mass",
                "--bc-unsafe-mass-target",
                "0.02",
                "--bc-max-unsafe-mass",
                "0.03",
                "--bc-safe-action-uniformity-weight",
                "1.5",
            ]
        )
        self.assertEqual(overridden.bc_safe_action_entropy_weight, 0.0)
        self.assertEqual(overridden.bc_min_safe_action_entropy, 0.97)
        self.assertEqual(overridden.bc_initialisation_objective, "safe_mass")
        self.assertEqual(overridden.bc_unsafe_mass_target, 0.02)
        self.assertEqual(overridden.bc_max_unsafe_mass, 0.03)
        self.assertEqual(overridden.bc_safe_action_uniformity_weight, 1.5)

    def test_verify_first_cli_defaults_false_and_accepts_true(self) -> None:
        self.assertEqual(build_parser().parse_args([]).verify_first, "false")
        self.assertEqual(
            build_parser().parse_args(["--verify-first", "true"]).verify_first,
            "true",
        )

    def test_frequency_cli_defaults_and_reaches_the_launcher(self) -> None:
        self.assertIsNone(build_parser().parse_args([]).freq)
        self.assertEqual(
            build_parser().parse_args(["--freq", "rollout"]).freq,
            "rollout",
        )

        env = build_launch_environment(
            environment="mini_pacman",
            seeds=[0],
            cpu_ids=[17],
            architecture="two_hidden",
            run_name="train_phase_run",
            n_iters=200,
            dry_run=True,
            adaptive_freq="rollout",
        )
        self.assertEqual(env["ADAPTIVE_FREQ"], "rollout")

        media_defaults = build_launch_environment(
            environment="media_streaming",
            seeds=[0],
            cpu_ids=[17],
            architecture="two_hidden",
            run_name="media_defaults",
            n_iters=200,
            dry_run=True,
        )
        self.assertEqual(media_defaults["ADAPTIVE_FREQ"], "1")
        self.assertEqual(media_defaults["TOTAL_TIMESTEPS"], "25000")

    def test_adaptive_granularity_cli_is_removed(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_launcher_args(["--adaptive-granularity", "train_phase"])


if __name__ == "__main__":
    unittest.main()
