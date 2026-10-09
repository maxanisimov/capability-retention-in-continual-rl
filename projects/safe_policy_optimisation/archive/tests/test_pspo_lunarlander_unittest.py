"""Tests for LunarLander support in the continuous-state PSPO stage."""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import tempfile
import unittest

import gymnasium as gym
import torch
from continuous_state_shields import LunarLanderDescentShield, MountainCarShield
from torch.utils.data import TensorDataset

from projects.safe_policy_optimisation.archive.scripts.build_lunarlander_descent_certificate import (
    descent_band_box,
    split_box,
)
from projects.safe_policy_optimisation.archive.stages.train_pspo_continuous import (
    LunarLanderDescentCostWrapper,
    parse_args,
    validate_base_architecture,
    validate_continuous_certificate,
    validate_lunarlander_certificate,
)


def _certificate(shield: LunarLanderDescentShield) -> TensorDataset:
    low, high = descent_band_box(shield.config)
    mask = torch.zeros((1, 4))
    mask[0, int(shield.config.main_engine_action)] = 1.0
    return TensorDataset(
        torch.as_tensor(low, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(high, dtype=torch.float32).unsqueeze(0),
        mask,
    )


class LunarLanderCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shield = LunarLanderDescentShield()

    def test_band_box_closes_the_open_runtime_conditions(self) -> None:
        low, high = descent_band_box(self.shield.config)
        self.assertAlmostEqual(high[1], self.shield.config.critical_height)
        self.assertAlmostEqual(high[3], self.shield.config.safe_min_vertical_speed)
        env = gym.make("LunarLander-v3", continuous=False)
        try:
            space_low = env.observation_space.low
            space_high = env.observation_space.high
        finally:
            env.close()
        # Every coordinate the shield does not read spans the whole space.
        for index in (0, 2, 4, 5, 6, 7):
            self.assertAlmostEqual(low[index], float(space_low[index]))
            self.assertAlmostEqual(high[index], float(space_high[index]))

    def test_matching_certificate_is_accepted(self) -> None:
        validate_lunarlander_certificate(_certificate(self.shield), self.shield)

    def test_split_boxes_tile_the_same_band(self) -> None:
        low, high = descent_band_box(self.shield.config)
        lows, highs = split_box(low, high, {0: 2, 4: 2})
        self.assertEqual(lows.shape, (4, 8))
        dataset = TensorDataset(
            torch.as_tensor(lows, dtype=torch.float32),
            torch.as_tensor(highs, dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]] * 4),
        )
        validate_lunarlander_certificate(dataset, self.shield)

    def test_mask_admitting_extra_actions_is_rejected(self) -> None:
        low, high = descent_band_box(self.shield.config)
        dataset = TensorDataset(
            torch.as_tensor(low, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(high, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        )
        with self.assertRaisesRegex(ValueError, "only safe action"):
            validate_lunarlander_certificate(dataset, self.shield)

    def test_box_wider_than_the_shield_band_is_rejected(self) -> None:
        low, high = descent_band_box(self.shield.config)
        high[3] = 0.5  # admits gentle descent the runtime shield never constrains
        dataset = TensorDataset(
            torch.as_tensor(low, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(high, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        )
        with self.assertRaisesRegex(ValueError, "vertical-speed upper bound"):
            validate_lunarlander_certificate(dataset, self.shield)

    def test_partial_band_coverage_is_rejected(self) -> None:
        low, high = descent_band_box(self.shield.config)
        low[0] = 0.0  # leaves the negative-x half of the band uncertified
        dataset = TensorDataset(
            torch.as_tensor(low, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(high, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        )
        with self.assertRaisesRegex(ValueError, "do not cover"):
            validate_lunarlander_certificate(dataset, self.shield)

    def test_mountaincar_shield_is_rejected_for_lunarlander(self) -> None:
        with self.assertRaisesRegex(ValueError, "lunarlander-descent shield"):
            validate_continuous_certificate(
                _certificate(self.shield), MountainCarShield(), "LunarLander-v3"
            )


class LunarLanderStageWiringTests(unittest.TestCase):
    def test_auto_shield_and_episode_cap_follow_the_environment(self) -> None:
        args = parse_args(
            [
                "--base-policy-path",
                "/tmp/base_policy.pt",
                "--env-id",
                "LunarLander-v3",
                "--mountaincar-shaped-reward",
                "false",
            ]
        )
        self.assertEqual(args.continuous_shield, "lunarlander-descent")
        self.assertEqual(args.max_episode_steps, 1000)
        self.assertEqual(args.env_id, "LunarLander-v3")

    def test_mountaincar_defaults_are_unchanged(self) -> None:
        args = parse_args(["--base-policy-path", "/tmp/base_policy.pt"])
        self.assertEqual(args.continuous_shield, "mountaincar")
        self.assertEqual(args.max_episode_steps, 200)
        self.assertTrue(args.mountaincar_shaped_reward)

    def test_shaped_reward_is_rejected_outside_mountaincar(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--base-policy-path",
                    "/tmp/base_policy.pt",
                    "--env-id",
                    "LunarLander-v3",
                ]
            )

    def test_mountaincar_shield_is_rejected_for_lunarlander(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--base-policy-path",
                    "/tmp/base_policy.pt",
                    "--env-id",
                    "LunarLander-v3",
                    "--continuous-shield",
                    "mountaincar",
                    "--mountaincar-shaped-reward",
                    "false",
                ]
            )

    def test_base_architecture_must_match_the_environment(self) -> None:
        architecture = {
            "input_dim": 2,
            "n_actions": 3,
            "state_representation": "continuous_features",
        }
        with self.assertRaisesRegex(ValueError, "LunarLander-v3"):
            validate_base_architecture(architecture, "LunarLander-v3")
        validate_base_architecture(architecture, "MountainCar-v0")


class LunarLanderCostWrapperTests(unittest.TestCase):
    def test_cost_marks_only_fast_descent_near_the_ground(self) -> None:
        env = gym.make("LunarLander-v3", continuous=False)
        try:
            wrapper = LunarLanderDescentCostWrapper(
                env, unsafe_height=0.5, unsafe_vertical_speed=-0.4, tolerance=1e-12
            )
            self.assertTrue(
                wrapper.is_unsafe([0.0, 0.4, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0])
            )
            self.assertFalse(
                wrapper.is_unsafe([0.0, 0.9, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0])
            )
            self.assertFalse(
                wrapper.is_unsafe([0.0, 0.4, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0])
            )
        finally:
            env.close()

    def test_the_unsafe_set_is_strictly_inside_the_shield_band(self) -> None:
        # The whole design rests on this gap; without it the shield could only
        # react to violations instead of preventing them.
        config = LunarLanderDescentShield().config
        args = parse_args(
            [
                "--base-policy-path",
                "/tmp/base_policy.pt",
                "--env-id",
                "LunarLander-v3",
                "--mountaincar-shaped-reward",
                "false",
            ]
        )
        self.assertLess(args.lunarlander_unsafe_height, config.critical_height)
        self.assertLess(
            args.lunarlander_unsafe_vertical_speed, config.safe_min_vertical_speed
        )


REPO_DIR = pathlib.Path(__file__).resolve().parents[3]
LAUNCHER = (
    REPO_DIR
    / "projects/safe_policy_optimisation/archive/scripts/run_pspo_lunarlander_descent.sh"
)


def _dry_run_trainer_argv(**overrides: str) -> list[str]:
    """The trainer command line the launcher would run, via DRY_RUN=1.

    RUN_ROOT goes to a temporary directory: the launcher writes its manifest
    there even under DRY_RUN, and a completed seed in the real RUN_ROOT would
    be skipped instead of printed.
    """
    with tempfile.TemporaryDirectory() as run_root:
        env = dict(
            os.environ,
            SEEDS="0",
            CPU_IDS="0",
            DRY_RUN="1",
            SKIP_EXISTING="1",
            RUN_ROOT=run_root,
            **overrides,
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_DIR,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    line = next(
        line
        for line in completed.stdout.splitlines()
        if line.strip().startswith("taskset")
    )
    argv = shlex.split(line)
    start = next(
        index
        for index, token in enumerate(argv)
        if token.endswith("train_pspo_continuous.py")
    )
    return argv[start + 1 :]


class LunarLanderLauncherTests(unittest.TestCase):
    """The launcher must be able to request a segment region.

    The segment parameters were parsed and then silently discarded before the
    2026-09-15 wiring, so an unplumbed launcher would run an orthotope arm while
    reporting a segment one. Both arms are checked here.
    """

    def test_the_launcher_defaults_to_the_orthotope_region(self) -> None:
        args = parse_args(_dry_run_trainer_argv())
        self.assertEqual(args.safe_region_shape, "orthotope")

    def test_the_launcher_plumbs_the_segment_region_and_its_parameters(self) -> None:
        argv = _dry_run_trainer_argv(
            SAFE_REGION_SHAPE="segment",
            SEGMENT_TOLERANCE="0.002",
            SEGMENT_SPLITS="6",
            SEGMENT_MAX_SPLITS="12",
        )
        args = parse_args(argv)
        self.assertEqual(args.safe_region_shape, "segment")
        self.assertAlmostEqual(args.segment_tolerance, 0.002)
        self.assertEqual(args.segment_splits, 6)
        self.assertEqual(args.segment_max_splits, 12)
        # The combinations the segment engine cannot honour must not be what the
        # launcher's own defaults produce.
        self.assertFalse(args.compute_region_once)
        self.assertNotEqual(args.region_update_mode, "union")
        self.assertEqual(args.growth_method, "IBP")


if __name__ == "__main__":
    unittest.main()
