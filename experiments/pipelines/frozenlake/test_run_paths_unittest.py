"""Unit tests for FrozenLake path resolution helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.pipelines.frozenlake.core.orchestration import run_paths


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


if __name__ == "__main__":
    unittest.main()
