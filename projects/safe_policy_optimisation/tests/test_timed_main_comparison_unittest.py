"""Output suppression and matched configuration checks for the timing sweep."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_timed_main_comparison as sweep


class TimingSweepTests(unittest.TestCase):
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
