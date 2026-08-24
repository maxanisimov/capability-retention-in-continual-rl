"""Tests for the multi-environment PSPO-adaptive launcher."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from projects.safe_policy_optimisation.scripts.launch_pspo_adaptive_multi_env import (
    DEFAULT_ENVS,
    build_launch_environment,
    build_parser,
    parse_launcher_args,
    parse_mpstat_idle,
    safety_demo_sizes,
    select_idle_cpus,
)


class PspoAdaptiveMultiEnvLauncherTests(unittest.TestCase):
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
        self.assertIn("--architecture", sections["policy initialisation:"])
        self.assertIn("--lid-n-iters", sections["LID settings:"])
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
                "colour_bomb": 74,
                "colour_bomb_v2": 856,
                "bridge_crossing": 332,
                "bridge_crossing_v2": 343,
                "mini_pacman": 8880,
            },
        )

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
        self.assertEqual(env["RASHOMON_BATCH_SIZE"], "auto")
        self.assertEqual(env["RASHOMON_CERTIFICATE_SAMPLES"], "all")
        self.assertEqual(env["RASHOMON_N_ITERS"], "200")
        self.assertEqual(env["BC_TARGET_MARGIN"], "2.0")
        self.assertEqual(env["DIRECTIONAL_RASHOMON_GROWTH"], "1")
        self.assertNotIn("ADAPTIVE_GRANULARITY", env)
        self.assertEqual(env["ADAPTIVE_FREQ"], "update")
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

    def test_frequency_cli_defaults_and_reaches_the_launcher(self) -> None:
        self.assertEqual(build_parser().parse_args([]).freq, "update")
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

    def test_adaptive_granularity_cli_is_removed(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_launcher_args(["--adaptive-granularity", "train_phase"])


if __name__ == "__main__":
    unittest.main()
