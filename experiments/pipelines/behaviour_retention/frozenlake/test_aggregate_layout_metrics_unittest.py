"""Unit tests for FrozenLake layout metric aggregation."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.behaviour_retention.frozenlake.core.analysis import aggregate_layout_metrics as alm


def _write_summary(path: Path, *, source_reward: float, downstream_reward: float, nested: bool) -> None:
    payload = {
        "source_mean_reward": source_reward,
        "downstream_mean_reward": downstream_reward,
    }
    if nested:
        payload = {"run_results": dict(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class FrozenLakeAggregateLayoutMetricsTests(unittest.TestCase):
    def test_reads_nested_and_legacy_flat_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            layout = "diagonal_test"
            _write_summary(
                outputs_root / layout / "seed_0" / "noadapt" / "run_summary.yaml",
                source_reward=1.0,
                downstream_reward=0.0,
                nested=True,
            )
            _write_summary(
                outputs_root / layout / "seed_1" / "noadapt" / "run_summary.yaml",
                source_reward=0.5,
                downstream_reward=0.5,
                nested=False,
            )

            with patch(
                "sys.argv",
                [
                    "aggregate_layout_metrics.py",
                    "--pipeline",
                    layout,
                    "--outputs-root",
                    str(outputs_root),
                ],
            ):
                alm.main()

            csv_path = outputs_root / layout / "aggregate_layout_metrics.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["policy"], "noadapt")
            self.assertEqual(rows[0]["source_total_reward_mean"], "0.750000")
            self.assertEqual(rows[0]["downstream_total_reward_mean"], "0.250000")


if __name__ == "__main__":
    unittest.main()

