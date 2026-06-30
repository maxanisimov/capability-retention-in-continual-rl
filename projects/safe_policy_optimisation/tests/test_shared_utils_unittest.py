"""Tests for the shared utility modules extracted during the library refactor."""

from __future__ import annotations

import argparse
import csv
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.safe_policy_optimisation.utils import io
from projects.safe_policy_optimisation.utils.cli import (
    PPO_HYPERPARAMETER_DEFAULTS,
    add_ppo_hyperparameter_args,
)
from projects.safe_policy_optimisation.utils.envs import (
    env_kwargs_from_args,
    parse_env_kwargs,
)
from projects.safe_policy_optimisation.utils.seeding import (
    EPISODE_SEED_OFFSET,
    EVAL_SEED_OFFSET,
    TRAIN_SEED_OFFSET,
    set_global_seeds,
)


class ParseEnvKwargsTests(unittest.TestCase):
    def test_none_returns_empty(self) -> None:
        self.assertEqual(parse_env_kwargs(None), {})

    def test_json_string_is_decoded(self) -> None:
        self.assertEqual(parse_env_kwargs('{"ghost_rand_prob": 0.25}'), {"ghost_rand_prob": 0.25})

    def test_dict_is_copied(self) -> None:
        original = {"a": 1}
        parsed = parse_env_kwargs(original)
        self.assertEqual(parsed, {"a": 1})
        self.assertIsNot(parsed, original)

    def test_non_object_json_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_env_kwargs("[1, 2, 3]")


class EnvKwargsFromArgsTests(unittest.TestCase):
    def test_explicit_kwargs_take_precedence(self) -> None:
        args = argparse.Namespace(env_kwargs='{"x": 1}', env_id="CustomMiniPacman-v0", ghost_rand_prob=0.5)
        self.assertEqual(env_kwargs_from_args(args), {"x": 1})

    def test_minipacman_default_ghost_prob(self) -> None:
        args = argparse.Namespace(env_kwargs=None, env_id="CustomMiniPacman-v0", ghost_rand_prob=0.3)
        self.assertEqual(env_kwargs_from_args(args), {"ghost_rand_prob": 0.3})

    def test_other_env_returns_empty(self) -> None:
        args = argparse.Namespace(env_kwargs=None, env_id="CustomBridgeCrossing-v0", ghost_rand_prob=0.3)
        self.assertEqual(env_kwargs_from_args(args), {})


class RecordIoTests(unittest.TestCase):
    def test_record_rows_adds_algorithm(self) -> None:
        rows = io.record_rows([{"episode": 0, "length": 3}], algorithm="plain_ppo")
        self.assertEqual(rows[0]["algorithm"], "plain_ppo")

    def test_record_training_rows_accumulates_end_timestep(self) -> None:
        rows = io.record_training_rows(
            [{"episode": 0, "length": 2}, {"episode": 1, "length": 3}], algorithm="shielded_ppo"
        )
        self.assertEqual([r["end_timestep"] for r in rows], [2, 5])
        self.assertTrue(all(r["algorithm"] == "shielded_ppo" for r in rows))

    def test_write_record_csv_header_and_end_timestep_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episodes.csv"
            io.write_record_csv(path, [], include_end_timestep=True)
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header[2], "end_timestep")
        self.assertIn("safe_trajectory", header)


class PpoArgsTests(unittest.TestCase):
    def test_defaults_match_shared_table(self) -> None:
        parser = add_ppo_hyperparameter_args(argparse.ArgumentParser())
        args = parser.parse_args([])
        for key, expected in PPO_HYPERPARAMETER_DEFAULTS.items():
            self.assertEqual(getattr(args, key), expected)


class SeedingTests(unittest.TestCase):
    def test_offsets_are_distinct(self) -> None:
        self.assertEqual(
            {TRAIN_SEED_OFFSET, EVAL_SEED_OFFSET, EPISODE_SEED_OFFSET},
            {30_000, 20_000, 10_000},
        )

    def test_set_global_seeds_is_reproducible(self) -> None:
        set_global_seeds(123)
        first = (random.random(), float(np.random.rand()))
        set_global_seeds(123)
        second = (random.random(), float(np.random.rand()))
        self.assertEqual(first, second)


class EvaluationMetricsTests(unittest.TestCase):
    def test_success_reward_safety_from_records(self) -> None:
        from projects.safe_policy_optimisation.utils.metrics import summarise_evaluation

        records = [
            {"reward": 1.0, "cost": 0.0, "length": 5, "violated": False, "safe_trajectory": True},
            {"reward": -1.0, "cost": 3.0, "length": 5, "violated": True, "safe_trajectory": False},
        ]
        m = summarise_evaluation(records, success_reward_threshold=0.0, cost_limit=0.0, algorithm="x")
        self.assertEqual(m["algorithm"], "x")
        self.assertEqual(m["eval_episodes"], 2)
        self.assertEqual(m["success"]["success_count"], 1)
        self.assertEqual(m["success"]["success_rate"], 0.5)
        self.assertEqual(m["reward"]["mean_total_reward"], 0.0)
        self.assertEqual(m["reward"]["max_total_reward"], 1.0)
        self.assertEqual(m["safety"]["violation_count"], 1)
        self.assertEqual(m["safety"]["safety_rate"], 0.5)
        self.assertEqual(m["safety"]["cost_limit"], 0.0)

    def test_accepts_episode_metrics_dataclass(self) -> None:
        from projects.safe_policy_optimisation.utils.metrics import summarise_evaluation
        from projects.safe_policy_optimisation.utils.safe_rl import EpisodeMetrics

        eps = [EpisodeMetrics(episode=0, reward=2.0, cost=0.0, length=3, violated=False)]
        m = summarise_evaluation(eps, success_reward_threshold=1.0)
        self.assertEqual(m["success"]["success_rate"], 1.0)
        self.assertIsNone(m["safety"]["cost_limit"])

    def test_empty_records(self) -> None:
        from projects.safe_policy_optimisation.utils.metrics import summarise_evaluation

        m = summarise_evaluation([], success_reward_threshold=0.0)
        self.assertEqual(m["eval_episodes"], 0)
        self.assertEqual(m["success"]["success_rate"], 0.0)


class ConfigSchemaTests(unittest.TestCase):
    def _valid_pipeline(self) -> dict:
        return {
            "description": "x",
            "module": "m",
            "default_task": "t",
            "output": {"output_dir": "o", "run_id": "r"},
            "runtime": {"seed": 0, "device": "cpu"},
            "training": {
                "total_timesteps": 2000,
                "learning_rate": 3e-4,
                "n_steps": 512,
                "batch_size": 128,
                "n_epochs": 4,
                "gamma": 0.99,
            },
        }

    def test_valid_pipeline_passes(self) -> None:
        from projects.safe_policy_optimisation.utils.config_schema import validate_pipeline_mapping

        cfg = validate_pipeline_mapping("p", self._valid_pipeline())
        self.assertEqual(cfg.default_task, "t")
        self.assertIn("training", cfg.sections)

    def test_unknown_section_rejected(self) -> None:
        from projects.safe_policy_optimisation.utils.config_schema import (
            SettingsValidationError,
            validate_pipeline_mapping,
        )

        bad = self._valid_pipeline()
        bad["trainning"] = {}
        with self.assertRaises(SettingsValidationError):
            validate_pipeline_mapping("p", bad)

    def test_unknown_field_rejected(self) -> None:
        from projects.safe_policy_optimisation.utils.config_schema import (
            SettingsValidationError,
            validate_pipeline_mapping,
        )

        bad = self._valid_pipeline()
        bad["training"]["lr"] = 0.1
        with self.assertRaises(SettingsValidationError):
            validate_pipeline_mapping("p", bad)

    def test_missing_required_field_rejected(self) -> None:
        from projects.safe_policy_optimisation.utils.config_schema import (
            SettingsValidationError,
            validate_pipeline_mapping,
        )

        bad = self._valid_pipeline()
        del bad["training"]["total_timesteps"]
        with self.assertRaises(SettingsValidationError):
            validate_pipeline_mapping("p", bad)

    def test_task_validation(self) -> None:
        from projects.safe_policy_optimisation.utils.config_schema import (
            SettingsValidationError,
            validate_task_mapping,
        )

        cfg = validate_task_mapping("t", {"env_id": "E-v0", "max_episode_steps": None, "env_kwargs": {"a": 1}})
        self.assertEqual(cfg.env_id, "E-v0")
        with self.assertRaises(SettingsValidationError):
            validate_task_mapping("t", {"env_id": "E-v0", "bogus": 1})


if __name__ == "__main__":
    unittest.main()
