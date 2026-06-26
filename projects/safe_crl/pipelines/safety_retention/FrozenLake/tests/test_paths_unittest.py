"""Unit tests for FrozenLake safety path helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from projects.safe_crl.pipelines.safety_retention.FrozenLake.core import paths


class FrozenLakeSafetyPathTests(unittest.TestCase):
    def test_mode_run_dirs_use_expected_subdirs_and_run_settings_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self.assertEqual(
                paths.mode_run_dir(root, "diagonal_4x4", "ppo", True, 3, "source"),
                root / "diagonal_4x4" / "ppo_deterministic" / "seed_3" / "noadapt",
            )
            self.assertEqual(
                paths.mode_run_dir(root, "diagonal_4x4", "ppo", True, 3, "downstream_rashomon"),
                root / "diagonal_4x4" / "ppo_deterministic" / "seed_3" / "downstream_rashomon",
            )

    def test_run_settings_tag_distinguishes_rl_and_dynamics(self) -> None:
        self.assertEqual(paths.run_settings_tag("ppo", True), "ppo_deterministic")
        self.assertEqual(paths.run_settings_tag("ppo", False), "ppo_stochastic")
        self.assertNotEqual(
            paths.run_settings_tag("ppo", True),
            paths.run_settings_tag("dqn", True),
        )

    def test_source_resolution_falls_back_to_legacy_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            legacy = root / "diagonal_4x4" / "ppo_deterministic" / "seed_0" / "source"
            legacy.mkdir(parents=True)

            self.assertEqual(
                paths.resolve_source_run_dir(root, "diagonal_4x4", "ppo", True, 0),
                legacy,
            )


if __name__ == "__main__":
    unittest.main()
