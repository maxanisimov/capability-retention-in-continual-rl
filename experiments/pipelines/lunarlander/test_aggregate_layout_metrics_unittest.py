"""Unit tests for aggregate layout metrics export."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.lunarlander.core.analysis import aggregate_layout_metrics as alm


def _write_run_summary(
    path: Path,
    *,
    source_reward: float,
    downstream_reward: float,
    source_success_rate: float = 0.0,
    downstream_success_rate: float = 0.0,
) -> None:
    payload = {
        "run_results": {
            "source_mean_reward": source_reward,
            "source_failure_rate": 0.0,
            "source_success_rate": source_success_rate,
            "downstream_mean_reward": downstream_reward,
            "downstream_failure_rate": 0.0,
            "downstream_success_rate": downstream_success_rate,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class AggregateLayoutMetricsTests(unittest.TestCase):
    def test_defaults_to_total_reward_and_success_without_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            task_setting = "unit_test_layout"

            _write_run_summary(
                outputs_root / task_setting / "seed_0" / "downstream_unconstrained" / "run_summary.yaml",
                source_reward=100.0,
                downstream_reward=130.0,
                source_success_rate=0.6,
                downstream_success_rate=0.8,
            )
            _write_run_summary(
                outputs_root / task_setting / "seed_1" / "downstream_unconstrained" / "run_summary.yaml",
                source_reward=120.0,
                downstream_reward=150.0,
                source_success_rate=0.4,
                downstream_success_rate=1.0,
            )

            with patch(
                "sys.argv",
                [
                    "aggregate_layout_metrics.py",
                    "--task-setting",
                    task_setting,
                    "--outputs-root",
                    str(outputs_root),
                ],
            ):
                alm.main()

            out_csv = outputs_root / task_setting / "aggregate_layout_metrics.csv"
            self.assertTrue(out_csv.exists())

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                rows = {row["policy"]: row for row in csv.DictReader(f)}

            unconstrained = rows["downstream_unconstrained"]

            self.assertIn("source_total_reward_mean", unconstrained)
            self.assertIn("downstream_total_reward_mean", unconstrained)
            self.assertIn("source_success_rate_mean", unconstrained)
            self.assertIn("downstream_success_rate_mean", unconstrained)
            self.assertNotIn("source_failure_rate_mean", unconstrained)
            self.assertNotIn("downstream_failure_rate_mean", unconstrained)
            self.assertNotIn("downstream_minus_source_total_reward_mean", unconstrained)

            self.assertEqual(unconstrained["source_total_reward_mean"], "110.000000")
            self.assertEqual(unconstrained["downstream_total_reward_mean"], "140.000000")
            self.assertEqual(unconstrained["source_success_rate_mean"], "0.500000")
            self.assertEqual(unconstrained["downstream_success_rate_mean"], "0.900000")

    def test_includes_relative_downstream_minus_source_reward_metric_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            task_setting = "unit_test_layout"

            # Policy A: deltas are +30 and +30 -> mean 30, std 0
            _write_run_summary(
                outputs_root / task_setting / "seed_0" / "downstream_unconstrained" / "run_summary.yaml",
                source_reward=100.0,
                downstream_reward=130.0,
            )
            _write_run_summary(
                outputs_root / task_setting / "seed_1" / "downstream_unconstrained" / "run_summary.yaml",
                source_reward=120.0,
                downstream_reward=150.0,
            )

            # Policy B: deltas are +20 and -20 -> mean 0, std 20 (population std)
            _write_run_summary(
                outputs_root / task_setting / "seed_0" / "downstream_rashomon" / "run_summary.yaml",
                source_reward=100.0,
                downstream_reward=120.0,
            )
            _write_run_summary(
                outputs_root / task_setting / "seed_1" / "downstream_rashomon" / "run_summary.yaml",
                source_reward=110.0,
                downstream_reward=90.0,
            )

            with patch(
                "sys.argv",
                [
                    "aggregate_layout_metrics.py",
                    "--task-setting",
                    task_setting,
                    "--outputs-root",
                    str(outputs_root),
                    "--compute-relative-to-source",
                ],
            ):
                alm.main()

            out_csv = outputs_root / task_setting / "aggregate_layout_metrics.csv"
            self.assertTrue(out_csv.exists())

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                rows = {row["policy"]: row for row in csv.DictReader(f)}

            self.assertIn("downstream_unconstrained", rows)
            self.assertIn("downstream_rashomon", rows)

            unconstrained = rows["downstream_unconstrained"]
            rashomon = rows["downstream_rashomon"]

            self.assertEqual(
                unconstrained["downstream_minus_source_total_reward_mean"],
                "30.000000",
            )
            self.assertEqual(
                unconstrained["downstream_minus_source_total_reward_std"],
                "0.000000",
            )
            self.assertEqual(
                rashomon["downstream_minus_source_total_reward_mean"],
                "0.000000",
            )
            self.assertEqual(
                rashomon["downstream_minus_source_total_reward_std"],
                "20.000000",
            )


if __name__ == "__main__":
    unittest.main()
