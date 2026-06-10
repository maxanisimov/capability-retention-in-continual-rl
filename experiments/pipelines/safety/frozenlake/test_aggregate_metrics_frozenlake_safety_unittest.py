"""Unit tests for FrozenLake shield safety metric aggregation."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import yaml

from experiments.pipelines.safety.frozenlake import aggregate_metrics_frozenlake_shield_safety as agg


def _write_summary(
    path: Path,
    *,
    source_safety_rate: float,
    source_trajectory_safety: float,
    source_reward: float,
    downstream_reward: float,
) -> None:
    payload = {
        "run_results": {
            "task_metrics": {
                "source": {
                    "safety_critical_state_safety_rate": source_safety_rate,
                    "greedy_trajectory_safety": source_trajectory_safety,
                    "total_reward": source_reward,
                },
                "downstream": {
                    "total_reward": downstream_reward,
                },
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class FrozenLakeSafetyAggregateMetricsTests(unittest.TestCase):
    def test_aggregates_defaults_and_writes_mean_pm_std_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            layout = "diagonal_4x4"
            _write_summary(
                outputs_root / layout / "seed_0" / "noadapt" / "run_summary.yaml",
                source_safety_rate=1.0,
                source_trajectory_safety=1.0,
                source_reward=1.0,
                downstream_reward=0.0,
            )
            _write_summary(
                outputs_root / layout / "seed_1" / "noadapt" / "run_summary.yaml",
                source_safety_rate=0.0,
                source_trajectory_safety=1.0,
                source_reward=0.0,
                downstream_reward=1.0,
            )

            rows = agg.aggregate_metrics(outputs_root=outputs_root, layout=layout)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].method, "noadapt")
            self.assertEqual(rows[0].num_seeds, 2)
            safety = rows[0].metrics["source_safety_critical_state_safety_rate"]
            self.assertEqual(safety.n, 2)
            self.assertAlmostEqual(float(safety.mean), 0.5)
            self.assertAlmostEqual(float(safety.std), 0.5)

            csv_path = outputs_root / layout / "aggregate.csv"
            tex_path = outputs_root / layout / "aggregate.tex"
            agg.write_csv(
                rows,
                csv_path,
                metric_keys=list(agg.DEFAULT_METRICS),
                precision=3,
            )
            agg.write_latex_table(
                rows,
                tex_path,
                metric_keys=list(agg.DEFAULT_METRICS),
                precision=3,
                layout=layout,
            )

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(
                csv_rows[0]["source_safety_critical_state_safety_rate"],
                "0.500 \u00b1 0.500",
            )
            self.assertEqual(
                csv_rows[0]["source_greedy_trajectory_safety"],
                "1.000 \u00b1 0.000",
            )

            latex = tex_path.read_text(encoding="utf-8")
            self.assertIn(r"$0.500 \pm 0.500$", latex)
            self.assertIn(r"\label{tab:frozenlake_shield_safety_diagonal_4x4_metrics}", latex)


if __name__ == "__main__":
    unittest.main()
