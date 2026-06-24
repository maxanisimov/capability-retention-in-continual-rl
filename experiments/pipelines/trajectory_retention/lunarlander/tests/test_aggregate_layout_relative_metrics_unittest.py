"""Unit tests for aggregate layout relative metrics export."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from experiments.pipelines.trajectory_retention.lunarlander.core.analysis import aggregate_layout_relative_metrics as alrm


def _write_run_summary(path: Path, run_results: dict[str, object]) -> None:
    payload = {"run_results": run_results}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class AggregateLayoutRelativeMetricsTests(unittest.TestCase):
    def test_defaults_to_total_reward_and_success_and_excludes_noadapt_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            task_setting = "unit_test_layout"

            # Baseline alias behavior: baseline available under "source" directories.
            _write_run_summary(
                outputs_root / task_setting / "seed_0" / "source" / "run_summary.yaml",
                {
                    "source_mean_reward": 100.0,
                    "source_success_rate": 0.70,
                    "source_failure_rate": 0.20,
                    "downstream_mean_reward": 80.0,
                    "downstream_success_rate": 0.40,
                    "downstream_failure_rate": 0.60,
                },
            )
            _write_run_summary(
                outputs_root / task_setting / "seed_1" / "source" / "run_summary.yaml",
                {
                    "source_mean_reward": 120.0,
                    "source_success_rate": 0.80,
                    "source_failure_rate": 0.10,
                    "downstream_mean_reward": 60.0,
                    "downstream_success_rate": 0.30,
                    "downstream_failure_rate": 0.70,
                },
            )

            _write_run_summary(
                outputs_root / task_setting / "seed_0" / "downstream_unconstrained" / "run_summary.yaml",
                {
                    "source_mean_reward": 110.0,
                    "source_success_rate": 0.75,
                    "source_failure_rate": 0.15,
                    "downstream_mean_reward": 95.0,
                    "downstream_success_rate": 0.50,
                    "downstream_failure_rate": 0.50,
                },
            )
            _write_run_summary(
                outputs_root / task_setting / "seed_1" / "downstream_unconstrained" / "run_summary.yaml",
                {
                    "source_mean_reward": 130.0,
                    "source_success_rate": 0.85,
                    "source_failure_rate": 0.05,
                    "downstream_mean_reward": 70.0,
                    "downstream_success_rate": 0.55,
                    "downstream_failure_rate": 0.45,
                },
            )

            _write_run_summary(
                outputs_root / task_setting / "seed_0" / "downstream_ewc" / "run_summary.yaml",
                {
                    "source_mean_reward": 105.0,
                    "source_success_rate": 0.72,
                    "source_failure_rate": 0.19,
                    "downstream_mean_reward": 90.0,
                    "downstream_success_rate": 0.45,
                    "downstream_failure_rate": 0.55,
                },
            )

            with patch(
                "sys.argv",
                [
                    "aggregate_layout_relative_metrics.py",
                    "--pipeline",
                    task_setting,
                    "--outputs-root",
                    str(outputs_root),
                ],
            ):
                alrm.main()

            out_csv = outputs_root / task_setting / "aggregate_layout_relative_metrics.csv"
            out_tex = outputs_root / task_setting / "aggregate_layout_relative_metrics.tex"
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_tex.exists())

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = {row["policy"]: row for row in reader}
                fieldnames = list(reader.fieldnames or [])

            self.assertEqual(set(rows.keys()), {"downstream_unconstrained", "downstream_ewc"})
            self.assertNotIn("noadapt", rows)
            self.assertNotIn("source", rows)

            unconstrained = rows["downstream_unconstrained"]
            self.assertEqual(unconstrained["num_seeds"], "2")

            self.assertEqual(unconstrained["relative_source_total_reward_mean"], "10.000000")
            self.assertEqual(unconstrained["relative_source_total_reward_std"], "0.000000")

            self.assertEqual(unconstrained["relative_source_success_rate_mean"], "0.050000")
            self.assertEqual(unconstrained["relative_source_success_rate_std"], "0.000000")

            self.assertEqual(unconstrained["relative_downstream_total_reward_mean"], "12.500000")
            self.assertEqual(unconstrained["relative_downstream_total_reward_std"], "2.500000")

            self.assertEqual(unconstrained["relative_downstream_success_rate_mean"], "0.175000")
            self.assertEqual(unconstrained["relative_downstream_success_rate_std"], "0.075000")

            self.assertNotIn("relative_source_failure_rate_mean", fieldnames)
            self.assertNotIn("relative_source_failure_rate_std", fieldnames)
            self.assertNotIn("relative_downstream_failure_rate_mean", fieldnames)
            self.assertNotIn("relative_downstream_failure_rate_std", fieldnames)

    def test_skips_missing_baseline_or_metric_pairs_and_limits_schema_to_selected_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir)
            task_setting = "unit_test_layout"

            _write_run_summary(
                outputs_root / task_setting / "seed_0" / "noadapt" / "run_summary.yaml",
                {
                    "source_mean_reward": 100.0,
                    "source_success_rate": 0.80,
                    "downstream_mean_reward": 80.0,
                    "downstream_success_rate": 0.40,
                },
            )

            # Seed 0 has a missing downstream_success_rate metric pair.
            _write_run_summary(
                outputs_root / task_setting / "seed_0" / "downstream_rashomon" / "run_summary.yaml",
                {
                    "source_mean_reward": 110.0,
                    "source_success_rate": 0.85,
                    "downstream_mean_reward": 81.0,
                },
            )
            # Seed 1 has no baseline and should be skipped.
            _write_run_summary(
                outputs_root / task_setting / "seed_1" / "downstream_rashomon" / "run_summary.yaml",
                {
                    "source_mean_reward": 111.0,
                    "source_success_rate": 0.86,
                    "downstream_mean_reward": 82.0,
                    "downstream_success_rate": 0.50,
                },
            )

            with patch(
                "sys.argv",
                [
                    "aggregate_layout_relative_metrics.py",
                    "--pipeline",
                    task_setting,
                    "--outputs-root",
                    str(outputs_root),
                    "--metric-groups",
                    "total_reward",
                    "success_rate",
                ],
            ):
                alrm.main()

            out_csv = outputs_root / task_setting / "aggregate_layout_relative_metrics.csv"
            self.assertTrue(out_csv.exists())

            with out_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = {row["policy"]: row for row in reader}
                fieldnames = list(reader.fieldnames or [])

            self.assertEqual(set(rows.keys()), {"downstream_rashomon"})
            rashomon = rows["downstream_rashomon"]
            self.assertEqual(rashomon["num_seeds"], "1")

            self.assertEqual(rashomon["relative_source_total_reward_mean"], "10.000000")
            self.assertEqual(rashomon["relative_downstream_total_reward_mean"], "1.000000")
            self.assertEqual(rashomon["relative_source_success_rate_mean"], "0.050000")
            self.assertEqual(rashomon["relative_downstream_success_rate_mean"], "")

            self.assertNotIn("relative_source_failure_rate_mean", fieldnames)
            self.assertNotIn("relative_downstream_failure_rate_mean", fieldnames)

            reserved = {"task_setting", "policy", "num_seeds"}
            for fieldname in fieldnames:
                if fieldname in reserved:
                    continue
                if fieldname.endswith("_mean") or fieldname.endswith("_std"):
                    self.assertTrue(fieldname.startswith("relative_"))


if __name__ == "__main__":
    unittest.main()
