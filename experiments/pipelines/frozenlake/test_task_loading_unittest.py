"""Unit tests for FrozenLake task-loading helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from experiments.pipelines.frozenlake.core.env.task_loading import load_task_settings


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class FrozenLakeTaskLoadingTests(unittest.TestCase):
    def test_split_pipeline_resolution_and_role_task_nums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            defs = root / "task_definitions.yaml"
            pipes = root / "task_pipelines.yaml"
            _write_yaml(
                defs,
                {
                    "src": {
                        "layout": "tiny",
                        "grid_size": 2,
                        "max_episode_steps": 8,
                        "env_map": ["SF", "HG"],
                    },
                    "dst": {
                        "layout": "tiny",
                        "grid_size": 2,
                        "max_episode_steps": 8,
                        "env_map": ["SF", "FG"],
                    },
                },
            )
            _write_yaml(
                pipes,
                {
                    "tiny_pipeline": {
                        "append_task_id": False,
                        "source": {"env": "src"},
                        "downstream": {"env": "dst"},
                    },
                },
            )

            source = load_task_settings(pipes, "tiny_pipeline", "source")
            downstream = load_task_settings(pipes, "tiny_pipeline", "downstream")

            self.assertEqual(source["task_num"], 0.0)
            self.assertEqual(downstream["task_num"], 1.0)
            self.assertFalse(source["append_task_id"])
            self.assertEqual(source["env_map"], ["SF", "HG"])
            self.assertEqual(downstream["_resolved_definition_name"], "dst")

    def test_legacy_direct_source_env_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "source_envs.yaml"
            _write_yaml(
                path,
                {
                    "diagonal_2x2": {
                        "grid_size": 2,
                        "max_episode_steps": 8,
                        "env1_map": ["SF", "FG"],
                    },
                },
            )

            cfg = load_task_settings(path, "diagonal_2x2", "source")

            self.assertEqual(cfg["task_num"], 0.0)
            self.assertEqual(cfg["env_map"], ["SF", "FG"])
            self.assertEqual(cfg["_task_settings_format"], "legacy_direct")


if __name__ == "__main__":
    unittest.main()

