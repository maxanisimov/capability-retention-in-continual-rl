"""Unit tests for task-loading schema split and role-fixed task-id behavior."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from experiments.pipelines.trajectory_retention.lunarlander.core.env.task_loading import load_task_settings


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class TaskLoadingSplitSchemaTests(unittest.TestCase):
    def test_split_pipeline_resolution_and_role_fixed_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            defs_path = root / "task_definitions.yaml"
            pipes_path = root / "task_pipelines.yaml"
            _write_yaml(
                defs_path,
                {
                    "env_src": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                        "gravity": -3.0,
                        "enable_wind": False,
                    },
                    "env_dst": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                        "gravity": -5.0,
                        "enable_wind": True,
                        "wind_power": 6.0,
                    },
                },
            )
            _write_yaml(
                pipes_path,
                {
                    "example_pipeline": {
                        "append_task_id": False,
                        "source": {"env": "env_src"},
                        "downstream": {"env": "env_dst"},
                    },
                },
            )

            src = load_task_settings(pipes_path, "example_pipeline", "source")
            dst = load_task_settings(pipes_path, "example_pipeline", "downstream")

            self.assertEqual(src["task_id"], 0.0)
            self.assertEqual(dst["task_id"], 1.0)
            self.assertFalse(src["append_task_id"])
            self.assertFalse(dst["append_task_id"])
            self.assertEqual(src["gravity"], -3.0)
            self.assertEqual(dst["gravity"], -5.0)
            self.assertEqual(src["_resolved_definition_name"], "env_src")
            self.assertEqual(dst["_resolved_definition_name"], "env_dst")
            self.assertEqual(src["_resolved_pipeline_name"], "example_pipeline")
            self.assertEqual(dst["_resolved_pipeline_name"], "example_pipeline")

    def test_split_pipeline_append_task_id_defaults_true_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            defs_path = root / "task_definitions.yaml"
            pipes_path = root / "task_pipelines.yaml"
            _write_yaml(
                defs_path,
                {
                    "env_src": {"env_id": "LunarLander-v4", "continuous": False},
                    "env_dst": {"env_id": "LunarLander-v4", "continuous": False},
                },
            )
            _write_yaml(
                pipes_path,
                {
                    "example_pipeline": {
                        "source": {"env": "env_src"},
                        "downstream": {"env": "env_dst"},
                    },
                },
            )

            src = load_task_settings(pipes_path, "example_pipeline", "source")
            dst = load_task_settings(pipes_path, "example_pipeline", "downstream")
            self.assertTrue(src["append_task_id"])
            self.assertTrue(dst["append_task_id"])

    def test_split_direct_definition_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            defs_path = root / "task_definitions.yaml"
            pipes_path = root / "task_pipelines.yaml"
            _write_yaml(
                defs_path,
                {
                    "direct_env": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                        "gravity": -4.0,
                    },
                    "fallback_env": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                        "gravity": -2.0,
                    },
                },
            )
            _write_yaml(
                pipes_path,
                {
                    "default": {
                        "append_task_id": True,
                        "source": {"env": "fallback_env"},
                        "downstream": {"env": "fallback_env"},
                    },
                },
            )

            cfg = load_task_settings(pipes_path, "direct_env", "source")
            self.assertEqual(cfg["env_id"], "LunarLander-v4")
            self.assertEqual(cfg["gravity"], -4.0)
            self.assertEqual(cfg["task_id"], 0.0)
            self.assertEqual(cfg["_resolved_definition_name"], "direct_env")
            self.assertIsNone(cfg["_resolved_pipeline_name"])

    def test_split_default_pipeline_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            defs_path = root / "task_definitions.yaml"
            pipes_path = root / "task_pipelines.yaml"
            _write_yaml(
                defs_path,
                {
                    "default_src": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                        "gravity": -3.0,
                    },
                    "default_dst": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                        "gravity": -7.0,
                    },
                },
            )
            _write_yaml(
                pipes_path,
                {
                    "default": {
                        "append_task_id": True,
                        "source": {"env": "default_src"},
                        "downstream": {"env": "default_dst"},
                    },
                },
            )

            cfg = load_task_settings(pipes_path, "missing_name", "downstream")
            self.assertEqual(cfg["_resolved_pipeline_name"], "default")
            self.assertEqual(cfg["_resolved_definition_name"], "default_dst")
            self.assertEqual(cfg["gravity"], -7.0)
            self.assertEqual(cfg["task_id"], 1.0)

    def test_split_missing_definition_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            defs_path = root / "task_definitions.yaml"
            pipes_path = root / "task_pipelines.yaml"
            _write_yaml(defs_path, {"present": {"env_id": "LunarLander-v4", "continuous": False}})
            _write_yaml(
                pipes_path,
                {
                    "broken": {
                        "append_task_id": True,
                        "source": {"env": "missing_def"},
                        "downstream": {"env": "present"},
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "not found"):
                load_task_settings(pipes_path, "broken", "source")

    def test_split_missing_pipeline_without_default_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            defs_path = root / "task_definitions.yaml"
            pipes_path = root / "task_pipelines.yaml"
            _write_yaml(
                defs_path,
                {
                    "env_src": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                    },
                    "env_dst": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                    },
                },
            )
            _write_yaml(
                pipes_path,
                {
                    "named_pipeline": {
                        "append_task_id": True,
                        "source": {"env": "env_src"},
                        "downstream": {"env": "env_dst"},
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "Task setting 'missing_pipeline' not found"):
                load_task_settings(pipes_path, "missing_pipeline", "source")

    def test_legacy_monolithic_is_still_supported_with_role_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_path = Path(tmp_dir) / "task_settings.yaml"
            _write_yaml(
                legacy_path,
                {
                    "legacy_case": {
                        "env_id": "LunarLander-v4",
                        "continuous": False,
                        "append_task_id": False,
                        "source": {
                            "task_id": 99.0,
                            "gravity": -1.0,
                            "enable_wind": False,
                        },
                        "downstream": {
                            "task_id": -7.0,
                            "gravity": -8.0,
                            "enable_wind": True,
                        },
                    },
                },
            )

            src = load_task_settings(legacy_path, "legacy_case", "source")
            dst = load_task_settings(legacy_path, "legacy_case", "downstream")
            self.assertEqual(src["task_id"], 0.0)
            self.assertEqual(dst["task_id"], 1.0)
            self.assertFalse(src["append_task_id"])
            self.assertFalse(dst["append_task_id"])
            self.assertEqual(src["gravity"], -1.0)
            self.assertEqual(dst["gravity"], -8.0)
            self.assertEqual(src["_task_settings_format"], "legacy_monolithic")

    def test_split_and_legacy_match_for_representative_pipeline(self) -> None:
        root = Path(__file__).resolve().parent
        legacy_path = root / "settings" / "tasks" / "task_settings.yaml"
        split_path = root / "settings" / "tasks" / "task_pipelines.yaml"
        legacy_all = yaml.safe_load(legacy_path.read_text(encoding="utf-8")) or {}
        split_all = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
        shared_pipelines = [name for name in split_all.keys() if name in legacy_all]
        if not shared_pipelines:
            self.skipTest("No shared pipelines found between legacy and split settings files.")
        pipeline = shared_pipelines[0]
        keys = [
            "env_id",
            "continuous",
            "gravity",
            "enable_wind",
            "wind_power",
            "turbulence_power",
            "initial_random_strength",
            "dispersion_strength",
            "main_engine_power",
            "side_engine_power",
            "leg_spring_torque",
            "lander_mass_scale",
            "leg_mass_scale",
            "linear_damping",
            "angular_damping",
            "terrain_heights",
            "action_repeat",
            "action_delay",
            "action_noise_prob",
            "action_noise_mode",
            "mark_out_of_viewport_as_unsafe",
            "append_task_id",
            "task_id",
        ]
        for role in ("source", "downstream"):
            legacy_cfg = load_task_settings(legacy_path, pipeline, role)
            split_cfg = load_task_settings(split_path, pipeline, role)
            for key in keys:
                self.assertEqual(
                    split_cfg.get(key),
                    legacy_cfg.get(key),
                    msg=f"Mismatch for {pipeline}:{role}:{key}",
                )


class RemovedTaskIdCliTests(unittest.TestCase):
    def test_source_train_does_not_declare_task_id_flag(self) -> None:
        script = Path(__file__).resolve().parent / "core" / "methods" / "source_train.py"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn('"--task-id"', text)

    def test_adapt_unconstrained_does_not_declare_source_downstream_task_id_flags(self) -> None:
        script = Path(__file__).resolve().parent / "core" / "methods" / "adapt_unconstrained.py"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn('"--source-task-id"', text)
        self.assertNotIn('"--downstream-task-id"', text)

    def test_adapt_ewc_does_not_declare_source_downstream_task_id_flags(self) -> None:
        script = Path(__file__).resolve().parent / "core" / "methods" / "adapt_ewc.py"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn('"--source-task-id"', text)
        self.assertNotIn('"--downstream-task-id"', text)

    def test_adapt_rashomon_does_not_declare_source_downstream_task_id_flags(self) -> None:
        script = Path(__file__).resolve().parent / "core" / "methods" / "adapt_rashomon.py"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn('"--source-task-id"', text)
        self.assertNotIn('"--downstream-task-id"', text)

    def test_adapt_rashomon_plus_does_not_declare_source_downstream_task_id_flags(self) -> None:
        script = Path(__file__).resolve().parent / "core" / "methods" / "adapt_rashomon_plus.py"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn('"--source-task-id"', text)
        self.assertNotIn('"--downstream-task-id"', text)

    def test_adapt_rashomon_expanded_does_not_declare_source_downstream_task_id_flags(self) -> None:
        script = Path(__file__).resolve().parent / "core" / "methods" / "adapt_rashomon_expanded.py"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn('"--source-task-id"', text)
        self.assertNotIn('"--downstream-task-id"', text)

    def test_expand_rashomon_set_does_not_declare_source_downstream_task_id_flags(self) -> None:
        script = Path(__file__).resolve().parent / "expand_rashomon_set.py"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn('"--source-task-id"', text)
        self.assertNotIn('"--downstream-task-id"', text)


if __name__ == "__main__":
    unittest.main()
