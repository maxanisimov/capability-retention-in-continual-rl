"""Parity tests against standard FrozenLake wrappers."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.pipelines.frozenlake.core.env.env_factory import make_env_from_layout as make_reference_env
from experiments.pipelines.frozenlake_safety.core.config import get_pipeline_config
from experiments.pipelines.frozenlake_safety.core.env import make_env as make_safety_env


class FrozenLakeSafetyWrapperParityTests(unittest.TestCase):
    def test_coordinate_observation_and_dense_shaping_match_reference_wrappers(self) -> None:
        cfg = get_pipeline_config("diagonal_4x4")
        actions = [2, 1, 1, 2, 2, 1]

        for shaped in (False, True):
            with self.subTest(shaped=shaped):
                reference_env = make_reference_env(
                    list(cfg.source_map),
                    cfg.max_episode_steps,
                    task_num=cfg.source_task_num,
                    shaped=shaped,
                )
                safety_env = make_safety_env(
                    cfg.source_map,
                    task_num=cfg.source_task_num,
                    max_episode_steps=cfg.max_episode_steps,
                    shaped=shaped,
                )
                try:
                    ref_obs, ref_info = reference_env.reset(seed=123)
                    safety_obs, safety_info = safety_env.reset(seed=123)
                    np.testing.assert_allclose(safety_obs, ref_obs)
                    self.assertEqual(safety_info["safe"], ref_info["safe"])

                    for action in actions:
                        ref_obs, ref_reward, ref_term, ref_trunc, ref_info = reference_env.step(action)
                        safety_obs, safety_reward, safety_term, safety_trunc, safety_info = safety_env.step(action)

                        np.testing.assert_allclose(safety_obs, ref_obs)
                        self.assertEqual(safety_reward, ref_reward)
                        self.assertEqual(safety_term, ref_term)
                        self.assertEqual(safety_trunc, ref_trunc)
                        self.assertEqual(safety_info["safe"], ref_info["safe"])
                        if ref_term or ref_trunc:
                            break
                finally:
                    reference_env.close()
                    safety_env.close()


if __name__ == "__main__":
    unittest.main()
