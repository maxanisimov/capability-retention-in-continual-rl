"""Output suppression and matched configuration checks for the timing sweep."""

from __future__ import annotations

import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import projects.safe_policy_optimisation.scripts.experiments.run_timed_main_comparison as sweep


class TimingSweepTests(unittest.TestCase):
    def test_all_baselines_have_360_jobs_without_pspo_initialisation(self):
        manifest = sweep.build_manifest(
            Path("/tmp/all-baseline-timing-test"), list(range(10)), None, False,
            methods=list(sweep.BASELINE_METHODS),
        )
        self.assertEqual(len(manifest["jobs"]), 360)
        self.assertEqual({job["method"] for job in manifest["jobs"]}, set(sweep.BASELINE_METHODS))
        self.assertTrue(all(job["kind"] == "training" for job in manifest["jobs"]))
        self.assertEqual(len({job["environment"] for job in manifest["jobs"]}), 6)
        for job in manifest["jobs"]:
            if job["method"] != "rl_sgf":
                continue
            values = sweep.training_values(job, Path("/tmp/rl-sgf-timing-test"), 0, False)
            self.assertEqual(values["init_policy_path"], job["config"]["init_policy_path"])
            self.assertEqual(values["cost_limit"], job["config"]["cost_limit"])
            self.assertEqual(values["rl_sgf_update_mode"], "ppo")

    def test_rl_sgf_new_manifests_and_legacy_jobs_keep_distinct_modes(self):
        for mode in ("ppo", "reinforce"):
            manifest = sweep.build_manifest(Path("/tmp/sgf-timing-mode-test"), [0],
                ["media_streaming"], False, methods=["rl_sgf"], rl_sgf_update_mode=mode)
            job = manifest["jobs"][0]
            self.assertEqual(job["rl_sgf_update_mode"], mode)
            self.assertEqual(sweep.training_values(job, Path("/tmp/sgf-run"), 0, False)["rl_sgf_update_mode"], mode)
            legacy = dict(job)
            del legacy["rl_sgf_update_mode"]
            self.assertEqual(sweep.rl_sgf_mode(legacy), "reinforce")
            self.assertEqual(sweep.training_values(legacy, Path("/tmp/sgf-run"), 0, False)["rl_sgf_update_mode"], "reinforce")

    def test_selecting_ppo_alias_does_not_create_pspo_jobs(self):
        manifest = sweep.build_manifest(
            Path("/tmp/ppo-timing-test"), [0], ["media_streaming"], False,
            methods=["ppo"],
        )
        self.assertEqual(len(manifest["jobs"]), 1)
        self.assertEqual(manifest["jobs"][0]["method"], "ppo_policy")
        with self.assertRaisesRegex(ValueError, "distinct supported"):
            sweep.build_manifest(Path("/tmp/duplicate-timing-test"), [0], None, False,
                                 methods=["ppo", "ppo_policy"])

    def test_csv_preserves_per_seed_timings_and_failure_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [dict(kind="training", environment="media_streaming", method="rl_sgf",
                         seed=seed, budget=25000, relative_dir=f"runs/seed{seed}")
                    for seed in (0, 1)]
            output = Path(tmp)
            sweep.write_json(output / jobs[0]["relative_dir"] / "training_time.json",
                             dict(status="complete", training_loop_s=1.5, process_wall_time_s=2.0))
            sweep.write_json(output / jobs[1]["relative_dir"] / "training_time.json",
                             dict(status="failed", error="test failure"))
            csv_path = sweep.write_training_time_csv(dict(output_dir=tmp, jobs=jobs))
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["training_loop_s"], "1.5")
            self.assertEqual(rows[0]["rl_sgf_update_mode"], "reinforce")
            self.assertEqual(rows[0]["method_label"], "RL-SGF-REINFORCE")
            self.assertEqual(rows[1]["status"], "failed")
            self.assertEqual(rows[1]["error"], "test failure")
            self.assertEqual(rows[1]["training_loop_s"], "")

    def test_manifest_has_420_independent_training_jobs_and_12_shared_initialisations(
        self,
    ):
        manifest = sweep.build_manifest(
            Path("/tmp/timing-test-manifest"), list(range(10)), None, False
        )
        jobs = manifest["jobs"]
        self.assertEqual(len(jobs), 432)
        self.assertEqual(len({job["id"] for job in jobs}), 432)
        self.assertEqual(sum(job["kind"] == "training" for job in jobs), 420)
        self.assertEqual(sum(job["kind"] == "initialisation" for job in jobs), 12)
        self.assertEqual(
            sum(
                job["method"] == "pspo_lookup" and job["kind"] == "training"
                for job in jobs
            ),
            60,
        )
        ids = {job["id"] for job in jobs}
        self.assertTrue(
            all(job["dependency"] in ids for job in jobs if "dependency" in job)
        )

    def test_all_methods_preserve_budget_hyperparameters_and_representation(self):
        manifest = sweep.build_manifest(
            Path("/tmp/timing-test-manifest"), [0], None, False
        )
        for job in manifest["jobs"]:
            if job["kind"] != "training":
                continue
            module = sweep.importlib.import_module(
                "projects.safe_policy_optimisation.stages."
                + sweep.STAGES[job["method"]]
            )
            parser = module.build_parser()
            values = sweep.training_values(job, Path("/tmp/test-run/seed0"), 1, False)
            argv = sweep.cli_arguments(parser, values)
            args = (
                module.parse_args(argv)
                if job["method"].startswith("pspo")
                else parser.parse_args(argv)
            )
            self.assertEqual(args.total_timesteps, job["budget"])
            hyperparameters = job["config"].get(
                "training_hyperparameters",
                job["config"].get("baseline_hyperparameters"),
            )
            for key, value in hyperparameters.items():
                self.assertEqual(getattr(args, key), value, (job["id"], key))
            self.assertEqual(args.curve_eval_freq, job["config"]["curve_eval_freq"])
            self.assertEqual(args.eval_episodes, 100)
            if job["method"].startswith("pspo"):
                self.assertEqual(args.state_representation, job["representation"])
                self.assertEqual(args.freq, job["config"]["adaptive"]["frequency"])
                self.assertFalse(args.verify_first)
            else:
                self.assertEqual(args.n_hidden, 2)

    def test_outputs_are_discarded_without_storing_then_deleting_metrics(self):
        from stable_baselines3.common.base_class import BaseAlgorithm

        from projects.safe_policy_optimisation.stages import train_ppo
        from projects.safe_policy_optimisation.utils import learning_curves

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.json"
            input_path.write_text('{"test_input": 1}')
            output = root / "outputs"
            output.mkdir()
            captured = {}
            with sweep.discard_stage_outputs(output, captured):
                self.assertEqual(json.loads(input_path.read_text())["test_input"], 1)
                train_ppo.write_json(
                    output / "metrics.json", {"reward": 123, "safety_rate": 1}
                )
                (output / "episodes.csv").write_text("reward,safety_rate\n123,1\n")
                logger = learning_curves.LearningCurveLogger(
                    curve_dir=output / "curves",
                    tensorboard_log_dir=output / "tensorboard",
                )
                logger.log_unshielded_evaluation(
                    timestep=8,
                    episode_rows=[
                        {
                            "episode": 0,
                            "total_reward": 123,
                            "success": True,
                            "length": 8,
                            "safe_trajectory": True,
                        }
                    ],
                )
                logger.close()
                BaseAlgorithm.save(None, output / "model.zip")
            self.assertEqual(captured["metrics"]["reward"], 123)
            self.assertFalse(any(path.is_file() for path in output.rglob("*")))

    def test_output_sink_does_not_buffer_training_rows(self):
        sink = sweep.NullTextIO()
        self.assertEqual(sink.write("x" * 1000000), 1000000)
        self.assertFalse(hasattr(sink, "getvalue"))


if __name__ == "__main__":
    unittest.main()
