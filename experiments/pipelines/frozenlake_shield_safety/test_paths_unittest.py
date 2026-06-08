"""Unit tests for FrozenLake shield safety path helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.pipelines.frozenlake_shield_safety.core import paths


class FrozenLakeSafetyPathTests(unittest.TestCase):
    def test_mode_run_dirs_use_expected_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self.assertEqual(
                paths.mode_run_dir(root, "diagonal_4x4", 3, "source"),
                root / "diagonal_4x4" / "seed_3" / "noadapt",
            )
            self.assertEqual(
                paths.mode_run_dir(root, "diagonal_4x4", 3, "downstream_rashomon"),
                root / "diagonal_4x4" / "seed_3" / "downstream_rashomon",
            )
            self.assertEqual(
                paths.mode_run_dir(root, "diagonal_4x4", 3, "downstream_safe_line_search"),
                root / "diagonal_4x4" / "seed_3" / "downstream_safe_line_search",
            )
            self.assertEqual(
                paths.mode_run_dir(root, "diagonal_4x4", 3, "downstream_lagrangian"),
                root / "diagonal_4x4" / "seed_3" / "downstream_lagrangian",
            )

    def test_source_resolution_falls_back_to_legacy_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            legacy = root / "diagonal_4x4" / "seed_0" / "source"
            legacy.mkdir(parents=True)

            self.assertEqual(paths.resolve_source_run_dir(root, "diagonal_4x4", 0), legacy)


if __name__ == "__main__":
    unittest.main()
