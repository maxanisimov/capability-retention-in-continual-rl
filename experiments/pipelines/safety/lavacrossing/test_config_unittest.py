"""Unit tests for LavaCrossing shield-safety configuration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments.pipelines.safety.lavacrossing.core.config import get_pipeline_config
from experiments.pipelines.safety.lavacrossing.core.env import make_env
from experiments.pipelines.safety.lavacrossing.core import pipeline
from experiments.pipelines.safety.lavacrossing.core.paths import mode_run_dir


PIPELINE_KEYS = (
    "corridor_7x7_deterministic",
    "corridor_7x7_slip_0p1",
    "route_switch_7x7_deterministic",
    "route_switch_7x7_slip_0p1",
)


class LavaCrossingConfigTests(unittest.TestCase):
    def test_all_initial_pipeline_keys_load(self) -> None:
        for key in PIPELINE_KEYS:
            with self.subTest(key=key):
                cfg = get_pipeline_config(key)

                self.assertEqual(cfg.layout, key)
                self.assertEqual(cfg.env_id, "CustomLavaCrossing-v0")
                self.assertEqual(cfg.n_actions, 5)
                self.assertEqual(len(cfg.source_map), 7)
                self.assertEqual(len(cfg.downstream_map), 7)
                self.assertTrue(all(len(row) == 7 for row in cfg.source_map))
                self.assertTrue(all(len(row) == 7 for row in cfg.downstream_map))
                if key.endswith("slip_0p1"):
                    self.assertEqual(cfg.dynamics, "stochastic")
                    self.assertAlmostEqual(cfg.slip_prob, 0.1)
                else:
                    self.assertEqual(cfg.dynamics, "deterministic")
                    self.assertAlmostEqual(cfg.slip_prob, 0.0)

    def test_env_construction_uses_coord_observation_and_five_actions(self) -> None:
        cfg = get_pipeline_config("corridor_7x7_slip_0p1")
        env = make_env(
            cfg.source_map,
            task_num=cfg.source_task_num,
            max_episode_steps=cfg.max_episode_steps,
            slip_prob=cfg.slip_prob,
        )
        try:
            obs, info = env.reset(seed=0)

            self.assertEqual(tuple(obs.shape), (3,))
            self.assertEqual(env.action_space.n, 5)
            self.assertAlmostEqual(float(env.unwrapped._slip_prob), 0.1)  # noqa: SLF001
            self.assertIn("safe", info)
            self.assertIn("is_success", info)
        finally:
            env.close()

    def test_artifact_paths_are_separated_by_pipeline_key(self) -> None:
        root = Path("/tmp/lavacrossing-test")
        det = mode_run_dir(root, "corridor_7x7_deterministic", 0, "source")
        slip = mode_run_dir(root, "corridor_7x7_slip_0p1", 0, "source")

        self.assertNotEqual(det, slip)
        self.assertIn("corridor_7x7_deterministic", str(det))
        self.assertIn("corridor_7x7_slip_0p1", str(slip))

    def test_run_experiment_dry_run_accepts_default_pipeline_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rc = pipeline.main(
                [
                    "--mode",
                    "source",
                    "--outputs-root",
                    tmp_dir,
                    "--dry-run",
                ],
            )

        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
