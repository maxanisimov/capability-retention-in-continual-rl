"""Tests for the policy-initialisation cluster launcher."""
from __future__ import annotations
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from projects.safe_policy_optimisation.scripts.lab_cluster import pspo_initialisation as cluster


class CpuAndSchedulerTests(unittest.TestCase):
    MPSTAT = """
Average: CPU %usr %idle
Average: all 1.0 99.0
Average: 0 0.0 100.0
Average: 1 1.0 99.0
Average: 2 5.0 95.0
Average: 3 20.0 80.0
Average: 7 2.0 98.0
"""

    def test_parser_threshold_and_reserved_low_cpus(self):
        result = cluster.ProbeResult("oak01", "ok", idle_by_cpu=cluster.parse_mpstat_idle(self.MPSTAT))
        self.assertEqual(result.eligible_cpus(minimum_idle=90, reserve=2), [2, 7])

    def test_probe_tsv_preserves_noncontiguous_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hosts.tsv"
            result = cluster.ProbeResult(
                "oak01", "ok", cores=8, mem_avail_gb=12, bitbucket="ok",
                source="ok", venv="ok", idle_by_cpu={0: 100, 2: 95, 6: 91})
            cluster.write_probe_tsv(path, [result], minimum_idle=90, reserve=2)
            with path.open() as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(row["idle_cpu_ids"], "2,6")
            hosts = cluster.read_hosts(
                path, mem_per_job_gb=2, minimum_idle=90, reserve=2)
            self.assertEqual(hosts[0].cpus, [2, 6])
            with self.assertRaisesRegex(ValueError, "differs from dispatch policy"):
                cluster.read_hosts(
                    path, mem_per_job_gb=2, minimum_idle=95, reserve=2)

    def test_memory_cap_spread_and_unique_slots(self):
        hosts = [cluster.Host("a", [2, 5], 10, 2),
                 cluster.Host("b", [3, 7], 3.9, 2),
                 cluster.Host("c", [4, 9], 10, 2)]
        units = [cluster.Unit("ce_only", "media_streaming", seed, "train") for seed in range(5)]
        self.assertEqual(cluster.spread(units, hosts), [])
        self.assertEqual([len(host.assignments) for host in hosts], [2, 1, 2])
        slots = [(host.name, item.cpu) for host in hosts for item in host.assignments]
        self.assertEqual(len(slots), len(set(slots)))

    def test_revalidation_rejects_busy_assigned_core(self):
        host = cluster.Host("oak01", [2], 10, 2)
        host.assignments.append(cluster.Assignment(
            cluster.Unit("ce_only", "media_streaming", 0, "train"), 2))
        fresh = cluster.ProbeResult("oak01", "ok", bitbucket="ok", source="ok",
                                    venv="ok", idle_by_cpu={2: 89.9})
        with mock.patch.object(cluster, "probe_host", return_value=fresh):
            errors = cluster.revalidate([host], domain="example", source_root=Path("/source"),
                                        minimum_idle=90, reserve=2, samples=1, timeout=1)
        self.assertIn("became busy", errors[0])


class DependencyAndCommandTests(unittest.TestCase):
    def write_base(self, root: Path, variant: str):
        directory = cluster.base_dir(root, variant, "media_streaming")
        directory.mkdir(parents=True)
        shield = root / "shield.pt"
        shield.write_bytes(b"shield")
        (directory / "base_policy.pt").write_bytes(b"policy")
        (directory / "safe_behaviour_dataset.pt").write_bytes(b"dataset")
        margin = 1.0 if variant == "no_entropy" else 0.0
        stopping = "target_margin" if variant == "no_entropy" else "any_safe_feasibility"
        mode = "all" if variant == "no_entropy" else "any"
        (directory / "summary.json").write_text(json.dumps({
            "base_policy_only": True, "shield_path": str(shield),
            "shield_sha256": cluster.sha256(shield),
            "architecture": {"n_hidden": 2, "hidden_dim": 64,
                             "state_representation": "one_hot_discrete_observation"},
            "dataset": {"dataset_size": 4},
            "base_policy": {"bc_margin_mode": mode, "initialisation_objective": "margin",
                            "margin_loss_weight": margin, "safe_action_entropy_weight": 0.0,
                            "stopping_criterion": stopping, "reached_target": True}}))

    def test_training_requires_base_and_resume_skips_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "training wave is blocked"):
                cluster.outstanding_units(phase="train", output_root=root,
                    variants=["ce_only"], environments=["media_streaming"], seeds=[0])
            self.write_base(root, "ce_only")
            seed0 = root / "ce_only/two_hidden/media_streaming/seed0"
            seed0.mkdir()
            (seed0 / "metrics.json").write_text("{}")
            units, _ = cluster.outstanding_units(phase="train", output_root=root,
                variants=["ce_only"], environments=["media_streaming"], seeds=[0, 1])
            self.assertEqual([unit.seed for unit in units], [1])

    def test_failed_initializer_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = cluster.base_failure_marker(root, "ce_only", "media_streaming")
            marker.parent.mkdir(parents=True)
            marker.write_text("rc=1")
            with self.assertRaisesRegex(ValueError, "previously failed"):
                cluster.outstanding_units(phase="base", output_root=root,
                    variants=["ce_only"], environments=["media_streaming"], seeds=[0])

    def test_base_and_train_commands_are_isolated(self):
        base = cluster.Assignment(cluster.Unit("no_entropy", "media_streaming", None, "base"), 11)
        base_env = cluster.command_environment(base, output_root=Path("/tmp/a"), smoke=False)
        self.assertEqual(base_env["PREPARE_BASE_ONLY"], "1")
        self.assertNotIn("BASE_POLICY_PATH", base_env)
        train = cluster.Assignment(cluster.Unit("ce_only", "bridge_crossing_v2", 7, "train"), 19)
        train_env = cluster.command_environment(train, output_root=Path("/tmp/a"), smoke=False)
        self.assertEqual(train_env["PREPARE_BASE_ONLY"], "0")
        self.assertEqual(train_env["CPU_IDS"], "19")
        self.assertEqual(train_env["TOTAL_TIMESTEPS"], "1600000")
        self.assertEqual(train_env["BC_MARGIN_LOSS_WEIGHT"], "0.0")

    @unittest.skipUnless(shutil.which("mpstat") and shutil.which("bash"), "needs mpstat and bash")
    def test_probe_remote_command_executes_and_parses_every_field(self):
        """The remote snippet must survive quoting; over-escaping silently zeroed the cluster."""
        source_root = Path(__file__).resolve().parents[3]

        def local_shell(host, remote, *, domain):
            return ["bash", "-lc", remote]

        with mock.patch.object(cluster, "ssh_command", local_shell):
            result = cluster.probe_host(
                "localhost", domain="invalid", source_root=source_root,
                samples=1, timeout=120,
            )
        self.assertEqual(result.status, "ok", msg=result.error)
        self.assertGreater(result.cores, 0)
        self.assertGreater(result.mem_avail_gb, 0.0)
        self.assertTrue(result.idle_by_cpu)

    def test_atomic_json_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            cluster.atomic_json(path, {"jobs": 120})
            self.assertEqual(json.loads(path.read_text()), {"jobs": 120})
            self.assertEqual(list(path.parent.glob(".manifest.json.*")), [])


if __name__ == "__main__":
    unittest.main()
