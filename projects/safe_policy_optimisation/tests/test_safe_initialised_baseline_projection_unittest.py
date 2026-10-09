from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from projects.safe_policy_optimisation.scripts import (
    run_safe_initialised_baseline_projection as experiment,
)


class SafeInitialisedBaselineProjectionTests(unittest.TestCase):
    def test_parse_mpstat_idle_uses_average_rows(self) -> None:
        output = """
12:00:01  2  1.0 2.0 97.0
Average: all 1.0 2.0 97.0
Average: 2 1.0 2.0 97.0
Average: 7 4.0 6.0 90.0
"""
        self.assertEqual(experiment.parse_mpstat_idle(output), {2: 97.0, 7: 90.0})

    def test_capacity_selects_only_complete_seed_blocks(self) -> None:
        self.assertEqual(
            experiment.select_methods_for_capacity(
                39, seeds_per_method=10, requested_methods=None
            ),
            ("ppo", "ppo_lagrangian", "ppo_pid_lagrangian"),
        )
        self.assertEqual(
            experiment.select_methods_for_capacity(
                40, seeds_per_method=10, requested_methods=None
            ),
            ("ppo", "ppo_lagrangian", "ppo_pid_lagrangian", "cpo"),
        )
        self.assertEqual(
            experiment.select_methods_for_capacity(
                50, seeds_per_method=10, requested_methods=None
            ),
            experiment.METHOD_PRIORITY,
        )
        self.assertEqual(
            experiment.select_methods_for_capacity(
                9, seeds_per_method=10, requested_methods=None
            ),
            (),
        )

    def test_explicit_methods_never_get_silently_truncated(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "need 20 free cores"):
            experiment.select_methods_for_capacity(
                19,
                seeds_per_method=10,
                requested_methods=("ppo", "cpo"),
            )

    @staticmethod
    def _architecture() -> dict[str, object]:
        return {
            "input_dim": 2,
            "n_actions": 2,
            "hidden_dim": 4,
            "n_hidden": 0,
            "activation": "Tanh",
            "state_representation": "one_hot_discrete_observation",
        }

    @staticmethod
    def _state(*, safe: bool, scale: float = 1.0) -> dict[str, torch.Tensor]:
        if safe:
            weight = torch.tensor([[0.0, 0.0], [scale, scale]])
        else:
            weight = torch.tensor([[scale, scale], [0.0, 0.0]])
        return {"0.weight": weight, "0.bias": torch.zeros(2)}

    def test_unsafe_actor_is_projected_and_reverified(self) -> None:
        architecture = self._architecture()
        policy = experiment.make_policy(architecture, self._state(safe=False))
        safe = self._state(safe=True)
        bounds = {
            "param_bounds_l": [safe["0.weight"], safe["0.bias"]],
            "param_bounds_u": [safe["0.weight"], safe["0.bias"]],
        }
        shield = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
        before, after, projection = experiment.audit_and_project(
            policy, architecture, bounds, shield
        )
        self.assertEqual(before["unsafe_states"], 2)
        self.assertEqual(after["unsafe_states"], 0)
        self.assertTrue(projection["applied"])
        self.assertTrue(projection["inside_lid_after"])
        self.assertGreater(projection["n_projected"], 0)

    def test_safe_actor_outside_lid_is_left_unchanged(self) -> None:
        architecture = self._architecture()
        state = self._state(safe=True, scale=2.0)
        policy = experiment.make_policy(architecture, state)
        original = {name: value.clone() for name, value in policy.state_dict().items()}
        lid = self._state(safe=True, scale=1.0)
        bounds = {
            "param_bounds_l": [lid["0.weight"], lid["0.bias"]],
            "param_bounds_u": [lid["0.weight"], lid["0.bias"]],
        }
        shield = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
        before, after, projection = experiment.audit_and_project(
            policy, architecture, bounds, shield
        )
        self.assertEqual(before["unsafe_states"], 0)
        self.assertEqual(after["unsafe_states"], 0)
        self.assertFalse(projection["applied"])
        self.assertFalse(projection["inside_lid_after"])
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_loads_custom_and_sb3_actor_layouts(self) -> None:
        architecture = self._architecture()
        expected = self._state(safe=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom = root / "cpo.pt"
            torch.save({"actor_state_dict": expected}, custom)
            loaded_custom = experiment.load_actor_state("cpo", custom, architecture)
            for name in expected:
                torch.testing.assert_close(loaded_custom[name], expected[name])

            mapping = experiment._base_to_ppo_actor_name_map(architecture)
            sb3_state = {mapping[name]: value for name, value in expected.items()}
            policy_file = root / "policy.pth"
            torch.save(sb3_state, policy_file)
            archive_path = root / "model.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(policy_file, "policy.pth")
            loaded_sb3 = experiment.load_actor_state("ppo", archive_path, architecture)
            for name in expected:
                torch.testing.assert_close(loaded_sb3[name], expected[name])

    def test_wave_jobs_have_unique_cpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = Namespace(
                seeds=[0, 1],
                output_root=Path(temporary),
                lid_root=Path("/lid"),
                smoke=True,
                force=False,
            )
            jobs = experiment.make_wave_jobs(
                args, "media_streaming", ("ppo", "cpo"), (10, 11, 12, 13)
            )
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({job.cpu for job in jobs}), 4)
        self.assertEqual(
            {(job.method, job.seed) for job in jobs},
            {("ppo", 0), ("ppo", 1), ("cpo", 0), ("cpo", 1)},
        )

    def test_report_excludes_incomplete_files(self) -> None:
        result = {
            "status": "complete",
            "smoke": False,
            "environment": "media_streaming",
            "method": "ppo",
            "seed": 0,
            "episodes": 2,
            "fixed_lid": {"matched_pspo_iterations": 10},
            "projection": {
                "applied": True,
                "inside_lid_before": False,
                "inside_lid_after": True,
                "n_projected": 2,
                "n_boundary": 2,
                "displacement_l2": 1.0,
                "displacement_linf": 1.0,
            },
            "raw": {
                "mean_total_reward": 1.0,
                "safe_trajectory_rate": 0.5,
                "exact_safety": {"alignment_rate": 0.5, "unsafe_states": 1},
            },
            "deployed": {
                "mean_total_reward": 2.0,
                "safe_trajectory_rate": 1.0,
                "exact_safety": {"alignment_rate": 1.0, "unsafe_states": 0},
            },
            "delta": {"mean_total_reward": 1.0, "safe_trajectory_rate": 0.5},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "media_streaming/ppo/seed0/postprocess_metrics.json"
            complete.parent.mkdir(parents=True)
            complete.write_text(json.dumps(result), encoding="utf-8")
            incomplete = root / "media_streaming/ppo/seed1/postprocess_metrics.json"
            incomplete.parent.mkdir(parents=True)
            incomplete.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            results = experiment.collect_results(root)
            experiment.write_report(root, results)
            aggregate = json.loads((root / "aggregate.json").read_text())
        self.assertEqual(len(results), 1)
        self.assertEqual(aggregate[0]["deployed_mean_total_reward"], 2.0)
        self.assertEqual(aggregate[0]["deployed_exact_unsafe_states"], 0)


if __name__ == "__main__":
    unittest.main()
