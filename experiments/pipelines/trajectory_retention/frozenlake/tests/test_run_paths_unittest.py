"""Unit tests for FrozenLake path resolution helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.pipelines.trajectory_retention.frozenlake.core.orchestration import run_paths


class FrozenLakeRunPathsTests(unittest.TestCase):
    def test_default_outputs_root_is_canonical_artifacts_runs(self) -> None:
        self.assertEqual(
            run_paths.default_outputs_root(),
            run_paths.runs_root(),
        )

    def test_source_resolution_prefers_noadapt_then_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            noadapt = root / "layout_a" / "seed_0" / "noadapt"
            source = root / "layout_a" / "seed_0" / "source"
            source.mkdir(parents=True)

            self.assertEqual(
                run_paths.resolve_default_source_run_dir(root, "layout_a", 0),
                source,
            )

            noadapt.mkdir()
            self.assertEqual(
                run_paths.resolve_default_source_run_dir(root, "layout_a", 0),
                noadapt,
            )

    def test_unconstrained_resolution_accepts_legacy_downstream_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            legacy = root / "layout_a" / "seed_1" / "downstream"
            legacy.mkdir(parents=True)

            self.assertEqual(
                run_paths.resolve_policy_dir(root, "layout_a", 1, "downstream_unconstrained"),
                legacy,
            )

    def test_seed_run_dir_is_tagged_by_rl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self.assertEqual(
                run_paths.seed_run_dir(root, "layout_a", 0),
                root / "layout_a" / "ppo" / "seed_0",
            )

    def test_source_resolution_prefers_tagged_path_over_pretag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pretag = root / "layout_a" / "seed_0" / "noadapt"
            pretag.mkdir(parents=True)
            tagged = root / "layout_a" / "ppo" / "seed_0" / "noadapt"
            tagged.mkdir(parents=True)

            self.assertEqual(
                run_paths.resolve_default_source_run_dir(root, "layout_a", 0),
                tagged,
            )

    def test_validate_rl_rejects_unsupported_algorithm(self) -> None:
        run_paths.validate_rl("ppo")
        with self.assertRaises(NotImplementedError):
            run_paths.validate_rl("dqn")


if __name__ == "__main__":
    unittest.main()
