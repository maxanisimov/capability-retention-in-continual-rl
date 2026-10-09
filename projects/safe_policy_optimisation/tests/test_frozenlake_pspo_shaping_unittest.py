"""PSPO reward shaping must not leak into initialization or final evaluation."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from projects.safe_policy_optimisation.scripts import (
    run_frozenlake_pspo_shaping as runner,
)


class FrozenLakePSPOShapingTests(unittest.TestCase):
    def test_safe_actor_is_prepared_without_constructing_a_potential(self):
        args = runner.build_parser().parse_args(
            ["--output-dir", "/tmp", "--size", "16"]
        )
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(runner, "FrozenLakePotentialReward") as shaping:
                _, mask, audit = runner.prepare_safety_only_actor(
                    args, Path(temp) / "inputs"
                )
                shaping.assert_not_called()
            self.assertEqual(mask.shape, (256, 4))
            actor = audit["initial_actor"]
            self.assertEqual(actor["initialisation_inputs"], ["shield_action_mask"])
            for key in (
                "reward_information_used",
                "goal_information_used",
                "witness_policy_used",
                "reward_training_used",
            ):
                self.assertFalse(actor[key])
            self.assertTrue(actor["safe_action_logits_equal"])
            self.assertTrue(audit["initial_actor_built_before_shaping"])
            self.assertFalse((Path(temp) / "inputs" / "potential.npz").exists())
            self.assertEqual(
                len((Path(temp) / "inputs/layout.txt").read_text().splitlines()), 16
            )

    def test_smoke_initial_actor_is_unchanged_and_final_policy_is_certified(self):
        with tempfile.TemporaryDirectory() as temp:
            args = runner.build_parser().parse_args(
                [
                    "--output-dir",
                    temp,
                    "--run-id",
                    "smoke",
                    "--size",
                    "16",
                    "--total-timesteps",
                    "64",
                    "--n-steps",
                    "32",
                    "--batch-size",
                    "16",
                    "--n-epochs",
                    "1",
                    "--max-episode-steps",
                    "16",
                    "--eval-episodes",
                    "2",
                    "--curve-eval-episodes",
                    "2",
                    "--curve-eval-freq",
                    "32",
                    "--lid-iters",
                    "1",
                    "--lid-checkpoint",
                    "1",
                    "--lid-batch-size",
                    "64",
                    "--verbose",
                    "0",
                ]
            )
            events = []
            finalize = runner.AdaptiveSafePPOV2.finalize_adaptive_update
            record = runner.FinalizedRewardCurve._evaluate_and_log

            def track_finalize(model):
                finalize(model)
                events.append(("finalize", model.num_timesteps))

            def track_record(callback, *, timestep):
                events.append(("evaluate", timestep))
                return record(callback, timestep=timestep)

            with (
                patch.object(
                    runner.AdaptiveSafePPOV2, "finalize_adaptive_update", track_finalize
                ),
                patch.object(
                    runner.FinalizedRewardCurve, "_evaluate_and_log", track_record
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = runner.run(args)
            root = Path(temp) / "smoke"
            config = json.loads((root / "config.json").read_text())
            init = config["initialization"]
            self.assertEqual(
                init["actor_sha256_before_shaping"], init["actor_sha256_after_shaping"]
            )
            self.assertEqual(summary["final_timesteps"], 64)
            self.assertEqual(summary["final_exact_all_state_alignment"], 1)
            self.assertEqual(summary["adaptive_diagnostics"]["final_flushes"], 1)
            self.assertLess(summary["max_discounted_telescoping_error"], 1e-10)
            self.assertGreater(summary["completed_training_episodes"], 0)
            self.assertFalse(summary["metrics"]["reward_shaping_enabled"])
            self.assertEqual(
                summary["metrics"]["evaluation_policy"], "greedy_unshielded"
            )
            self.assertEqual(events[-2:], [("finalize", 64), ("evaluate", 64)])
            self.assertEqual(events.count(("evaluate", 64)), 1)
            certificate = json.loads((root / "certificate.json").read_text())
            self.assertFalse(certificate["certificate_sampling"])
            self.assertEqual(certificate["greedy_alignment"], 1)
            self.assertEqual(
                certificate["safety_winning_states_checked"],
                init["safety_winning_states"],
            )
            with (root / "episodes.csv").open() as handle:
                episodes = list(csv.DictReader(handle))
            for row in episodes:
                self.assertAlmostEqual(
                    float(row["total_reward"]),
                    float(row["success"] == "True") - 0.001 * int(row["length"]),
                )
                self.assertEqual(row["safe_trajectory"], "True")
            with (root / "training_episodes.csv").open() as handle:
                training = list(csv.DictReader(handle))
            self.assertTrue(training)
            for row in training:
                self.assertLess(abs(float(row["telescoping_error"])), 1e-10)


if __name__ == "__main__":
    unittest.main()
