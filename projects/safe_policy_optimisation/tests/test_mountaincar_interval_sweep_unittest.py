"""Tests for controlled, detached MountainCar shield interval sweeps."""

from __future__ import annotations

import fcntl
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from continuous_state_shields import MountainCarShield, MountainCarShieldConfig

from projects.safe_policy_optimisation.scripts import (
    run_mountaincar_shield_interval_sweep as sweep,
)
from projects.safe_policy_optimisation.stages.train_pspo_continuous import (
    parse_args,
    validate_mountaincar_certificate,
)


class MountainCarIntervalSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "sweep"
        self.record = sweep.prepare(self.root, smoke=False)

    def complete(self, pair: dict, reward: float, safety: float) -> Path:
        run = self.root / pair["id"]
        (run / "initialisation").mkdir(parents=True)
        (run / "training").mkdir()
        (run / "initialisation/base_policy.pt").write_bytes(b"synthetic-test-policy")
        sweep.write_json(run / "initialisation/summary.json", {"final_verification": {"all_certified": True}})
        sweep.write_json(run / "training/config.json", {
            "continuous_shield_config": sweep.shield_config(pair["upper_bound"]),
            "base_policy_sha256": sweep.sha256(run / "initialisation/base_policy.pt"),
        })
        sweep.write_json(run / "training/summary.json", {
            "final_interval_certified": True, "evaluation_policy": "unshielded",
            "final_timesteps": 400384, "early_stop_triggered": False,
            "trajectory_safety": {"safe_trajectory_rate": safety},
            "evaluation_proposed_action_safety": {"proposed_action_checks": 1000, "unsafe_proposed_action_count": 0},
            "timing": {"training_wall_time_s": 1.0},
        })
        sweep.write_json(run / "training/metrics.json", {
            "eval_episodes": 100, "reward": {"mean_total_reward": reward},
            "safety": {"safety_rate": safety},
        })
        result = sweep.validate_result(run, pair, self.record["settings"])
        sweep.write_json(run / "result.json", result)
        sweep.write_json(run / "status.json", {"status": "complete"})
        return run

    def test_all_pairs_and_distinct_initialisations(self) -> None:
        pairs = self.record["pairs"]
        self.assertEqual(len(pairs), 60)
        self.assertEqual(len({pair["id"] for pair in pairs}), 60)
        self.assertEqual({pair["seed"] for pair in pairs}, set(range(10)))
        for pair in pairs:
            commands = sweep.build_commands(self.root, pair, self.record["settings"])
            args = parse_args(commands[1][3:])
            self.assertEqual(args.total_timesteps, 400000)
            self.assertEqual(args.adaptive_granularity, "train_phase")
            self.assertEqual(args.rashomon_n_iters, 200)
            self.assertEqual(args.evaluation_policy, "unshielded")
            self.assertEqual(args.early_stop_eval_freq, 0)
            self.assertEqual(args.rashomon_objective, "weighted_width")
            self.assertIn(pair["id"], str(args.base_policy_path))

    def test_full_certificates_and_velocity_rule(self) -> None:
        for upper in sweep.UPPER_BOUNDS:
            shield = MountainCarShield(MountainCarShieldConfig(**sweep.shield_config(upper)))
            path = self.root / "_inputs" / sweep.interval_name(upper) / "critical_interval_dataset.pt"
            dataset = torch.load(path, weights_only=False)
            validate_mountaincar_certificate(dataset, shield)
            low, high, mask = dataset.tensors
            np.testing.assert_allclose(low.numpy(), [[-1.2, -0.07]], atol=1e-7)
            np.testing.assert_allclose(high.numpy(), [[upper, 0]], atol=1e-7)
            self.assertEqual(mask.tolist(), [[False, False, True]])
            for x in (-1.2, (-1.2 + upper) / 2, upper):
                self.assertEqual(shield.get_safe_actions([x, -0.01]), [2])
                self.assertEqual(shield.get_safe_actions([x, 0.0]), [0, 1, 2])
                self.assertEqual(shield.get_safe_actions([x, 0.01]), [0, 1, 2])
            self.assertEqual(shield.get_safe_actions([np.float32(-1.2), -0.01]), [2])
            self.assertEqual(shield.get_safe_actions([upper + 0.001, -0.01]), [0, 1, 2])

    def test_physical_core_selection_and_busy_siblings(self) -> None:
        topology = "# CPU,CORE,SOCKET,ONLINE\n0,0,0,Y\n1,1,0,Y\n2,2,0,Y\n3,3,0,Y\n4,2,0,Y\n5,3,0,Y\n6,4,0,Y\n7,5,0,N\n"
        idle = {cpu: 100.0 for cpu in range(8)}
        allowed = set(range(8))
        self.assertEqual(sweep.select_idle_physical_cpus(topology, idle, allowed, set(), 95), [2, 3, 6])
        self.assertEqual(sweep.select_idle_physical_cpus(topology, idle, allowed, {4}, 95), [3, 6])
        idle[5] = 94.9
        self.assertEqual(sweep.select_idle_physical_cpus(topology, idle, allowed, set(), 95), [2, 6])
        self.assertEqual(sweep.select_idle_physical_cpus(topology, idle, {4}, set(), 95), [4])

    def test_queue_does_not_overwrite_failed_or_existing_pairs(self) -> None:
        pairs = self.record["pairs"]
        run = self.root / pairs[0]["id"]
        run.mkdir(parents=True)
        sweep.write_json(run / "status.json", {"status": "failed"})
        self.assertEqual(len(sweep.pending_pairs(self.root, pairs)), 59)
        self.assertNotIn(pairs[0], sweep.pending_pairs(self.root, pairs))
        with self.assertRaises(FileExistsError):
            sweep.prepare(self.root, smoke=False)

    def test_duplicate_controller_lock(self) -> None:
        with (self.root / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                sweep.controller(self.root, 95, 60)

    def test_controller_dispatches_queue_and_continues_after_failure(self) -> None:
        record = self.record.copy()
        record["pairs"] = record["pairs"][:3]
        sweep.write_json(self.root / "experiment.json", record)
        launched = []

        class FakeProcess:
            def __init__(self, pid: int, code: int):
                self.pid, self.code, self.polls = pid, code, 0

            def poll(self) -> int | None:
                self.polls += 1
                return None if self.polls == 1 else self.code

        def launch(command: list[str], **_kwargs) -> FakeProcess:
            pair_id = command[command.index("--worker") + 1]
            pair = next(pair for pair in record["pairs"] if pair["id"] == pair_id)
            launched.append(pair_id)
            if pair == record["pairs"][1]:
                run = self.root / pair_id
                run.mkdir(parents=True)
                sweep.write_json(run / "status.json", {"status": "failed", "error": "synthetic failure"})
                return FakeProcess(len(launched), 1)
            self.complete(pair, -190, 0.8)
            return FakeProcess(len(launched), 0)

        def cpus(busy: set[int], _min_idle: float) -> tuple[list[int], dict]:
            available = sorted({8, 9} - busy)
            return available, {"admitted_cpus": available}

        with patch.object(sweep.subprocess, "Popen", side_effect=launch), \
                patch.object(sweep, "free_physical_cpus", side_effect=cpus), \
                patch.object(sweep.time, "sleep"):
            sweep.controller(self.root, 95, 2)
        self.assertEqual(launched, [pair["id"] for pair in record["pairs"]])
        manifest = sweep.read_json(self.root / "launch_manifest.json")
        self.assertEqual([job["cpu"] for job in manifest["jobs"]], [8, 9, 8])
        self.assertEqual(manifest["failed_pairs"], [record["pairs"][1]["id"]])
        self.assertEqual(sweep.read_json(self.root / "aggregate.json")["completed_pairs"], 2)

    def test_aggregate_completed_only_and_uncertainty(self) -> None:
        for pair in self.record["pairs"]:
            if pair["seed"] in (0, 1):
                self.complete(pair, -200 + 20 * pair["seed"], 0.5 + 0.5 * pair["seed"])
        failed = self.record["pairs"][12]
        run = self.root / failed["id"]
        run.mkdir(parents=True)
        sweep.write_json(run / "status.json", {"status": "failed"})
        aggregate = sweep.write_report(self.root)
        self.assertEqual(aggregate["completed_pairs"], 12)
        self.assertEqual(aggregate["failed_pairs"], [failed["id"]])
        for interval in aggregate["intervals"]:
            self.assertEqual(interval["completed_seeds"], 2)
            self.assertEqual(interval["mean_total_reward"], -190)
            self.assertTrue(math.isclose(interval["mean_total_reward_2se"], 20))
            self.assertEqual(interval["safety_rate"], 0.75)
            self.assertEqual(interval["action_compliance"], 1)

    def test_refuses_bad_certificate_or_disagreeing_wall_audit(self) -> None:
        pair = self.record["pairs"][0]
        run = self.complete(pair, -190, 0.8)
        summary = sweep.read_json(run / "training/summary.json")
        summary["final_interval_certified"] = False
        sweep.write_json(run / "training/summary.json", summary)
        with self.assertRaisesRegex(RuntimeError, "certificate"):
            sweep.validate_result(run, pair, self.record["settings"])
        summary["final_interval_certified"] = True
        summary["trajectory_safety"]["safe_trajectory_rate"] = 0.9
        sweep.write_json(run / "training/summary.json", summary)
        with self.assertRaisesRegex(RuntimeError, "audits disagree"):
            sweep.validate_result(run, pair, self.record["settings"])

    def test_input_provenance_guard(self) -> None:
        sweep.verify_provenance(self.root, self.record)
        name = next(iter(self.record["input_sha256"]))
        (self.root / name).write_bytes(b"modified-test-input")
        with self.assertRaisesRegex(RuntimeError, "input changed"):
            sweep.verify_provenance(self.root, self.record)


if __name__ == "__main__":
    unittest.main()
