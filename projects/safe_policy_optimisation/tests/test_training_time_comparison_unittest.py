"""Timing boundaries, retry handling, and efficient TensorBoard extraction."""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_training_time_comparison as timing


def record(event):
    data = event.SerializeToString()
    length = struct.pack("<Q", len(data))
    return (
        length
        + struct.pack("<I", timing.masked_crc32c(length))
        + data
        + struct.pack("<I", timing.masked_crc32c(data))
    )


class TrainingTimeTests(unittest.TestCase):
    def setUp(self):
        timing.read_json.cache_clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def events(self, name, final_step=100, many=False):
        path = self.root / name
        with path.open("wb") as handle:
            handle.write(
                record(timing.Event(wall_time=1000, file_version="brain.Event:2"))
            )
            if many:
                for step in range(5000):
                    handle.write(
                        record(timing.Event(wall_time=1001 + step / 5000, step=step))
                    )
            handle.write(record(timing.Event(wall_time=1015, step=final_step)))
        return path

    def test_event_endpoints_read_large_suffix(self):
        path = self.events("events.out.tfevents.1.host.1", final_step=5000, many=True)
        result = timing.event_endpoints(path)
        self.assertGreater(path.stat().st_size, 65536)
        self.assertEqual(result["last_step"], 5000)
        self.assertEqual(result["seconds"], 15)
        self.assertEqual(result["host"], "host")

    def test_truncated_or_corrupted_suffix_is_not_repaired(self):
        path = self.events("events")
        data = path.read_bytes()
        path.write_bytes(data[:-2])
        with self.assertRaises(ValueError):
            timing.event_endpoints(path)
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 255]))
        with self.assertRaises(ValueError):
            timing.event_endpoints(path)

    def test_retries_align_final_steps_and_do_not_merge(self):
        incomplete = self.events("failed", final_step=50)
        complete = self.events("success")
        stream, status, _ = timing.select_event_stream([incomplete, complete], 100)
        self.assertEqual(status, "partial_logged_interval")
        self.assertEqual(stream["seconds"], 15)
        duplicate = self.events("duplicate")
        stream, status, _ = timing.select_event_stream([complete, duplicate], 100)
        self.assertIsNone(stream)
        self.assertEqual(status, "ambiguous_completed_streams")

    def test_only_successes_count_and_duplicates_stay_visible(self):
        path = self.root / "log"
        path.write_text(
            "[1/3] failed media_streaming/seed0/ppo (9s)\n"
            "[2/3] ok media_streaming/seed0/ppo (12s)\n"
            "[3/3] ok other/seed0/ppo (15s)\n"
            "[4/4] ok media_streaming/seed0/ppo (13s)\n"
        )
        env = SimpleNamespace(key="media_streaming")
        with patch.object(timing, "launcher_paths", return_value=[path]):
            records = timing.read_completions(env, False)
        self.assertEqual([item["seconds"] for item in records[(0, "ppo")]], [12, 13])
        self.assertEqual(len(records), 1)

    def test_wrong_extended_budget_is_rejected(self):
        path = self.root / "log"
        path.write_text("launching jobs smoke=1600000\n")
        env = SimpleNamespace(key="mini_pacman", nominal_budget=2000000)
        with patch.object(timing, "launcher_paths", return_value=[path]):
            with self.assertRaises(ValueError):
                timing.read_completions(env, False)

    def fake_run(self, key, adaptive=False):
        relative = "metrics.json" if adaptive else f"{key}/metrics.json"
        method = SimpleNamespace(
            key=key,
            label=key,
            relative_path=relative,
            adaptive=adaptive,
            section=(key,) if key == "ppo_lagrangian" else (),
        )
        env = SimpleNamespace(
            key="media_streaming",
            label="Media Streaming",
            nominal_budget=100,
            baseline_root=self.root,
            adaptive_root=self.root,
        )
        directory = self.root / "seed0" / Path(relative).parent
        directory.mkdir(parents=True)
        config = {
            "seed": 0,
            "total_timesteps": {key: 100} if key == "ppo_lagrangian" else 100,
        }
        if adaptive:
            config["base_policy_path"] = str(self.root / "base" / "base_policy.pt")
        summary = {"final_timesteps": 100}
        if key == "ppo_lagrangian":
            summary = {key: summary}
        metrics = (
            {key: {"eval_episodes": 100}}
            if key == "ppo_lagrangian"
            else {"eval_episodes": 100}
        )
        for name, payload in (
            ("config.json", config),
            ("summary.json", summary),
            ("metrics.json", metrics),
        ):
            (directory / name).write_text(json.dumps(payload))
        return env, method

    def test_combined_lag_time_is_not_assigned_to_individual_method(self):
        env, method = self.fake_run("ppo_lagrangian")
        records = {
            (0, "baselines_lag"): [
                {"seconds": 12, "source": "log:1", "host": None, "core": None}
            ]
        }
        row = timing.build_seed_row(env, method, 0, records)
        self.assertIsNone(row["rl_process_s"])
        self.assertIsNone(row["total_s"])
        self.assertEqual(row["combined_lag_pid_process_s"], 12)

    def test_missing_shared_initialisation_prevents_pspo_total(self):
        env, method = self.fake_run("pspo", adaptive=True)
        records = {
            (0, "pspo"): [{"seconds": 12, "source": "log:1", "host": None, "core": "4"}]
        }
        row = timing.build_seed_row(env, method, 0, records)
        self.assertEqual(row["rl_process_s"], 12)
        self.assertIsNone(row["initialisation_s"])
        self.assertIsNone(row["total_s"])
        self.assertIsNone(row["lid_s"])

    def test_no_double_counting_nested_overhead(self):
        env, method = self.fake_run("pspo", adaptive=True)
        directory = self.root / "seed0"
        (directory / "training_time.json").write_text(
            json.dumps({"policy_initialisation_s": 20, "rl_training_s": 9})
        )
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "final_timesteps": 100,
                    "adaptive_diagnostics": {
                        "rashomon_wall_time_total_s": 3,
                        "projection_wall_time_total_s": 1,
                    },
                }
            )
        )
        records = {
            (0, "pspo"): [{"seconds": 12, "source": "log:1", "host": None, "core": "4"}]
        }
        row = timing.build_seed_row(env, method, 0, records)
        self.assertEqual(row["total_s"], 32)
        self.assertEqual(row["training_loop_s"], 9)
        self.assertEqual(row["lid_s"], 3)

    def test_missing_singleton_and_sample_standard_error(self):
        self.assertEqual(
            timing.aggregate_values([]), {"n": 0, "mean_s": None, "two_se_s": None}
        )
        self.assertIsNone(timing.aggregate_values([12])["two_se_s"])
        stats = timing.aggregate_values([10, 20, 30])
        self.assertEqual(stats["mean_s"], 20)
        self.assertAlmostEqual(stats["two_se_s"], 20 / math.sqrt(3))
        self.assertEqual(timing.seconds(0), 0)
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                timing.seconds(value)

    def test_parent_pipeline_stage_is_separate_and_must_match_selected_run(self):
        env, method = self.fake_run("ppo_policy")
        directory = self.root / "seed0" / "ppo_policy"
        summary = json.loads((directory / "summary.json").read_text())
        parent = {
            "total_timesteps_budget": 100,
            "stages": {
                "ppo_policy": {
                    "run_dir": str(directory),
                    "summary": summary,
                    "started_at": 1000,
                    "finished_at": 1010,
                    "elapsed_seconds": 10,
                    "cpu_ids": [4],
                }
            },
        }
        (directory.parent / "summary.json").write_text(json.dumps(parent))
        records = {
            (0, "ppo"): [{"seconds": 15, "source": "log:1", "host": None, "core": None}]
        }
        row = timing.build_seed_row(env, method, 0, records)
        self.assertEqual(row["pipeline_stage_s"], 10)
        self.assertEqual(row["rl_process_s"], 15)
        self.assertEqual(row["total_s"], 15)
        self.assertIsNone(row["rl_stage_s"])
        parent["stages"]["ppo_policy"]["summary"]["final_timesteps"] = 50
        (directory.parent / "summary.json").write_text(json.dumps(parent))
        timing.read_json.cache_clear()
        row = timing.build_seed_row(env, method, 0, records)
        self.assertIsNone(row["pipeline_stage_s"])
        self.assertEqual(
            row["pipeline_stage_status"], "record_does_not_match_selected_run"
        )

    def test_joint_pipeline_stage_is_not_split(self):
        env, method = self.fake_run("ppo_lagrangian")
        directory = self.root / "seed0" / "ppo_lagrangian"
        summary = json.loads((directory / "summary.json").read_text())
        parent = {
            "total_timesteps_budget": 100,
            "stages": {
                "ppo_lagrangian": {
                    "run_dir": str(directory),
                    "summary": summary,
                    "started_at": 1000,
                    "finished_at": 1010,
                    "elapsed_seconds": 10,
                }
            },
        }
        (directory.parent / "summary.json").write_text(json.dumps(parent))
        row = timing.build_seed_row(env, method, 0, {})
        self.assertIsNone(row["pipeline_stage_s"])
        self.assertEqual(row["combined_lag_pid_pipeline_stage_s"], 10)

    def test_parent_stage_must_cover_the_selected_event_interval(self):
        directory = self.root / "seed0" / "ppo_policy"
        directory.mkdir(parents=True)
        parent = {
            "total_timesteps_budget": 100,
            "stages": {
                "ppo_policy": {
                    "run_dir": str(directory),
                    "summary": {},
                    "started_at": 1000,
                    "finished_at": 1010,
                    "elapsed_seconds": 10,
                }
            },
        }
        (directory.parent / "summary.json").write_text(json.dumps(parent))
        row = {
            "nominal_timesteps": 100,
            "logged_start_wall_time": 1001,
            "logged_end_wall_time": 1015,
            "pipeline_stage_s": None,
        }
        timing.add_pipeline_timing(
            row, directory, SimpleNamespace(key="ppo_policy", adaptive=False), {}, {}
        )
        self.assertIsNone(row["pipeline_stage_s"])
        self.assertEqual(
            row["pipeline_stage_status"], "invalid_or_unaligned_stage_interval"
        )


if __name__ == "__main__":
    unittest.main()
