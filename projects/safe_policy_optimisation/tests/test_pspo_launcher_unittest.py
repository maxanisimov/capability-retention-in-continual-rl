"""Tests for PSPO launcher setting and cache validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from projects.safe_policy_optimisation.utils.pspo_launcher import (
    base_policy_artifact_matches,
    initial_safe_set_matches,
    parse_cpu_ids,
    resolve_certificate_samples,
    resolve_seed_cpu_ids,
    resolve_target_margin,
    resolve_total_timesteps,
)
from projects.safe_policy_optimisation.utils.rashomon import (
    parse_rashomon_batch_size,
    resolve_rashomon_batch_size,
)


class PspoLauncherTests(unittest.TestCase):
    def test_auto_rashomon_batch_uses_complete_safe_behaviour_dataset(self) -> None:
        mask = np.asarray([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=np.float32)

        self.assertEqual(resolve_rashomon_batch_size("auto", mask), 3)
        self.assertEqual(resolve_rashomon_batch_size(2, mask), 2)
        self.assertEqual(parse_rashomon_batch_size("all"), "auto")
        self.assertEqual(parse_rashomon_batch_size("17"), 17)

    def test_explicit_and_legacy_cpu_allocations(self) -> None:
        self.assertEqual(parse_cpu_ids("3,7-9 12"), [3, 7, 8, 9, 12])
        self.assertEqual(
            resolve_seed_cpu_ids([0, 1, 2], cpu_ids="4,8,11"),
            [4, 8, 11],
        )
        self.assertEqual(
            resolve_seed_cpu_ids([0, 1, 2], core_start="20"),
            [20, 21, 22],
        )
        with self.assertRaises(ValueError):
            resolve_seed_cpu_ids([0, 1], cpu_ids="4")

    def test_exhaustive_certificates_count_only_states_with_safe_actions(self) -> None:
        mask = np.asarray([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=np.float32)

        self.assertEqual(
            resolve_certificate_samples("all", default=1000, shield_mask=mask),
            3,
        )

    def test_certificate_samples_reject_invalid_values(self) -> None:
        for value in ("invalid", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_certificate_samples(value, default=1000)

    def test_target_margin_override_and_validation(self) -> None:
        self.assertEqual(resolve_target_margin("0.1", default=10.0), 0.1)
        self.assertEqual(resolve_target_margin("", default=2.0), 2.0)
        for value in ("-0.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_target_margin(value, default=1.0)

    def test_total_timesteps_override_and_validation(self) -> None:
        self.assertEqual(resolve_total_timesteps("400000", default=200_000), 400_000)
        self.assertEqual(resolve_total_timesteps("", default=200_000), 200_000)
        for value in ("invalid", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_total_timesteps(value, default=200_000)

    def test_initial_set_cache_requires_every_requested_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            safe_set = Path(tmp)
            (safe_set / "base_policy.pt").write_bytes(b"policy")
            (safe_set / "rashomon_param_bounds.pt").write_bytes(b"bounds")
            (safe_set / "summary.json").write_text(
                json.dumps(
                    {
                        "base_policy": {
                            "bc_margin_mode": "all",
                            "target_margin": 0.1,
                        },
                        "rashomon": {
                            "multi_label_mode": "all",
                            "surrogate": "logsumexp",
                            "certificate_samples": 441,
                            "n_iters": 200,
                        },
                    }
                ),
                encoding="utf-8",
            )

            settings = {
                "multi_label_mode": "all",
                "surrogate": "logsumexp",
                "target_margin": 0.1,
                "certificate_samples": 441,
                "n_iters": 200,
            }
            self.assertTrue(initial_safe_set_matches(safe_set, **settings))
            for name, value in (
                ("multi_label_mode", "any"),
                ("surrogate", "auto"),
                ("target_margin", 0.2),
                ("certificate_samples", 440),
                ("n_iters", 201),
            ):
                with self.subTest(name=name):
                    mismatch = dict(settings)
                    mismatch[name] = value
                    self.assertFalse(initial_safe_set_matches(safe_set, **mismatch))

            entropy_settings = {
                **settings,
                "safe_action_entropy_weight": 1.0,
                "min_safe_action_entropy": 0.95,
            }
            self.assertFalse(initial_safe_set_matches(safe_set, **entropy_settings))
            summary = json.loads((safe_set / "summary.json").read_text())
            summary["base_policy"].update(
                {
                    "safe_action_entropy_weight": 1.0,
                    "min_safe_action_entropy": 0.95,
                }
            )
            (safe_set / "summary.json").write_text(json.dumps(summary))
            self.assertTrue(initial_safe_set_matches(safe_set, **entropy_settings))
            entropy_settings["min_safe_action_entropy"] = 0.9
            self.assertFalse(initial_safe_set_matches(safe_set, **entropy_settings))

            summary = json.loads((safe_set / "summary.json").read_text())
            summary["base_policy"].update(
                {
                    "initialisation_objective": "safe_mass",
                    "unsafe_mass_target": 0.01,
                    "max_unsafe_mass": 0.02,
                    "safe_action_uniformity_weight": 1.0,
                    "safe_action_entropy_weight": 0.0,
                    "min_safe_action_entropy": 0.95,
                }
            )
            (safe_set / "summary.json").write_text(json.dumps(summary))
            safe_mass_settings = {
                **settings,
                "target_margin": 999.0,
                "initialisation_objective": "safe_mass",
                "unsafe_mass_target": 0.01,
                "max_unsafe_mass": 0.02,
                "safe_action_uniformity_weight": 1.0,
                "min_safe_action_entropy": 0.95,
            }
            self.assertTrue(initial_safe_set_matches(safe_set, **safe_mass_settings))
            safe_mass_settings["unsafe_mass_target"] = 0.015
            self.assertFalse(initial_safe_set_matches(safe_set, **safe_mass_settings))

    def test_base_policy_only_skips_rashomon_and_writes_reusable_artifacts(self) -> None:
        from projects.safe_policy_optimisation.stages import compute_shield_rashomon_set

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shield_path = root / "shield_q.pt"
            torch.save(
                {
                    "shield": np.asarray(
                        [[1, 0], [1, 1], [0, 1]],
                        dtype=np.int64,
                    )
                },
                shield_path,
            )
            args = compute_shield_rashomon_set.build_parser().parse_args(
                [
                    "--shield-path",
                    str(shield_path),
                    "--output-dir",
                    str(root),
                    "--run-id",
                    "base",
                    "--state-representation",
                    "one_hot",
                    "--n-hidden",
                    "0",
                    "--bc-margin-mode",
                    "all",
                    "--bc-target-margin",
                    "2.0",
                    "--linear-init-margin",
                    "2.0",
                    "--base-policy-only",
                ]
            )

            with mock.patch.object(
                compute_shield_rashomon_set,
                "calibrate_inverse_temperature",
                side_effect=AssertionError("base-only mode must not calibrate"),
            ):
                summary = compute_shield_rashomon_set.run(args)

            base_dir = root / "base"
            self.assertTrue(summary["base_policy_only"])
            self.assertEqual(summary["rashomon"]["status"], "skipped")
            self.assertFalse((base_dir / "rashomon_param_bounds.pt").exists())
            self.assertTrue(
                base_policy_artifact_matches(
                    base_dir,
                    shield_path=shield_path,
                    dataset_size=3,
                    hidden_dim=64,
                    n_hidden=0,
                    state_representation="one_hot_discrete_observation",
                    margin_mode="all",
                    target_margin=2.0,
                    safe_action_entropy_weight=1.0,
                    min_safe_action_entropy=0.95,
                )
            )
            self.assertFalse(
                base_policy_artifact_matches(
                    base_dir,
                    shield_path=shield_path,
                    dataset_size=3,
                    hidden_dim=64,
                    n_hidden=0,
                    state_representation="one_hot_discrete_observation",
                    margin_mode="all",
                    target_margin=2.0,
                    safe_action_entropy_weight=0.0,
                    min_safe_action_entropy=0.95,
                )
            )

    def test_entropy_base_policy_only_writes_reusable_artifacts(self) -> None:
        from projects.safe_policy_optimisation.stages import compute_shield_rashomon_set

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shield_path = root / "shield_q.pt"
            torch.save(
                {
                    "shield": np.asarray(
                        [[1, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.int64
                    )
                },
                shield_path,
            )
            args = compute_shield_rashomon_set.build_parser().parse_args(
                [
                    "--shield-path",
                    str(shield_path),
                    "--output-dir",
                    str(root),
                    "--run-id",
                    "entropy_base",
                    "--state-representation",
                    "one_hot",
                    "--n-hidden",
                    "0",
                    "--bc-margin-mode",
                    "all",
                    "--bc-target-margin",
                    "2.0",
                    "--linear-init-margin",
                    "2.0",
                    "--bc-safe-action-entropy-weight",
                    "1.0",
                    "--bc-min-safe-action-entropy",
                    "0.95",
                    "--base-policy-only",
                ]
            )

            summary = compute_shield_rashomon_set.run(args)
            base_dir = root / "entropy_base"

            self.assertEqual(
                summary["base_policy"]["final_normalized_safe_action_entropy_min"],
                1.0,
            )
            self.assertTrue(
                base_policy_artifact_matches(
                    base_dir,
                    shield_path=shield_path,
                    dataset_size=3,
                    hidden_dim=64,
                    n_hidden=0,
                    state_representation="one_hot_discrete_observation",
                    margin_mode="all",
                    target_margin=2.0,
                    safe_action_entropy_weight=1.0,
                    min_safe_action_entropy=0.95,
                )
            )


if __name__ == "__main__":
    unittest.main()
