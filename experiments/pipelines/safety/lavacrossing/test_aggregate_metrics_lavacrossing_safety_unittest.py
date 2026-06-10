"""Unit tests for LavaCrossing shield-safety metric aggregation."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import yaml

from experiments.pipelines.safety.lavacrossing import aggregate_metrics_lavacrossing_shield_safety as agg


def _write_summary(path: Path) -> None:
    payload = {
        "run_settings": {
            "env_id": "CustomLavaCrossing-v0",
            "pipeline_key": "corridor_7x7_slip_0p1",
            "dynamics": "stochastic",
            "slip_prob": 0.1,
            "n_actions": 5,
            "max_episode_steps": 100,
            "shield_type": "probabilistic",
        },
        "run_results": {
            "source_success_rate": 0.9,
            "downstream_success_rate": 0.8,
            "source_failure_rate": 0.1,
            "downstream_failure_rate": 0.2,
            "task_metrics": {
                "source": {
                    "safety_critical_state_safety_rate": 1.0,
                    "greedy_trajectory_safety": 1.0,
                    "total_reward": 0.9,
                },
                "downstream": {
                    "total_reward": 0.8,
                },
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class LavaCrossingAggregateMetricsTests(unittest.TestCase):
    def test_csv_includes_environment_setup_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            layout = "corridor_7x7_slip_0p1"
            _write_summary(outputs_root / layout / "seed_0" / "noadapt" / "run_summary.yaml")

            rows = agg.aggregate_metrics(outputs_root=outputs_root, layout=layout)
            self.assertEqual(rows[0].env_setup["dynamics"], "stochastic")
            self.assertEqual(rows[0].env_setup["slip_prob"], 0.1)

            csv_path = outputs_root / layout / "aggregate.csv"
            agg.write_csv(
                rows,
                csv_path,
                metric_keys=list(agg.DEFAULT_METRICS),
                precision=3,
            )

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(csv_rows[0]["env_id"], "CustomLavaCrossing-v0")
            self.assertEqual(csv_rows[0]["dynamics"], "stochastic")
            self.assertEqual(csv_rows[0]["slip_prob"], "0.1")
            self.assertEqual(csv_rows[0]["n_actions"], "5")
            self.assertEqual(csv_rows[0]["shield_type"], "probabilistic")


if __name__ == "__main__":
    unittest.main()
