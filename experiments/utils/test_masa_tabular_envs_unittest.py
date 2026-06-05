"""Unit tests for custom MASA-style tabular environments."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import gymnasium as gym

    import experiments.utils.masa_tabular_envs  # noqa: F401
    from experiments.utils.masa_tabular_envs.factory import make_custom_masa_env
    from experiments.utils.masa_tabular_envs.gridworlds import CustomColourBombGridWorld, CustomColourGridWorld
    from experiments.utils.masa_tabular_envs.media_streaming import CustomMediaStreaming
    from experiments.utils.masa_tabular_envs.pacman import CustomMiniPacman, CustomPacman
    from experiments.utils.masa_tabular_envs.renderers.colour_bomb_grid_world import ColourBombGridWorldRenderer
except ModuleNotFoundError:  # pragma: no cover - exercised only without RL extras
    gym = None


@unittest.skipIf(gym is None, "gymnasium is not installed")
class CustomMasaTabularEnvTests(unittest.TestCase):
    ENV_IDS = [
        "CustomBridgeCrossing-v0",
        "CustomBridgeCrossingV2-v0",
        "CustomColourGridWorld-v0",
        "CustomColourBombGridWorld-v0",
        "CustomColourBombGridWorldV2-v0",
        "CustomColourBombGridWorldV3-v0",
        "CustomMediaStreaming-v0",
        "CustomMediaStreamingV2-v0",
        "CustomMediaStreamingV3-v0",
        "CustomMiniPacman-v0",
        "CustomPacman-v0",
    ]

    def test_registered_envs_reset_step_and_close(self) -> None:
        for env_id in self.ENV_IDS:
            with self.subTest(env_id=env_id):
                env = gym.make(env_id)
                try:
                    obs, info = env.reset(seed=123)
                    self.assertTrue(env.observation_space.contains(obs))
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    self.assertTrue(env.observation_space.contains(next_obs))
                    self.assertIsInstance(float(reward), float)
                    self.assertIsInstance(terminated, bool)
                    self.assertIsInstance(truncated, bool)
                finally:
                    env.close()

    def test_factory_passes_env_kwargs_and_time_limit(self) -> None:
        env = make_custom_masa_env(
            "CustomColourGridWorld-v0",
            max_episode_steps=7,
            env_kwargs={"slip_prob": 0.12},
        )
        try:
            self.assertAlmostEqual(env.unwrapped._slip_prob, 0.12)  # noqa: SLF001
            self.assertEqual(env.spec.max_episode_steps, 7)
        finally:
            env.close()

    def test_slip_probability_controls_grid_transitions(self) -> None:
        deterministic_env = CustomColourGridWorld(slip_prob=0.0)
        stochastic_env = CustomColourGridWorld(slip_prob=0.2)
        try:
            deterministic_matrix = deterministic_env.get_transition_matrix()
            stochastic_matrix = stochastic_env.get_transition_matrix()
            self.assertEqual(int(np.argmax(deterministic_matrix[:, 0, 1])), 1)
            self.assertAlmostEqual(float(deterministic_matrix[:, 0, 1].sum()), 1.0)
            self.assertAlmostEqual(float(stochastic_matrix[:, 0, 1].sum()), 1.0)
            self.assertFalse(np.allclose(deterministic_matrix[:, 0, 1], stochastic_matrix[:, 0, 1]))
        finally:
            deterministic_env.close()
            stochastic_env.close()

    def test_media_streaming_rates_change_transition_matrix(self) -> None:
        slow = CustomMediaStreaming(fast_rate=0.6)
        fast = CustomMediaStreaming(fast_rate=0.95)
        try:
            self.assertFalse(np.allclose(slow.get_transition_matrix(), fast.get_transition_matrix()))
            sums = fast.get_transition_matrix().sum(axis=0)
            self.assertTrue(np.allclose(sums, 1.0))
        finally:
            slow.close()
            fast.close()

    def test_pacman_ghost_randomness_changes_distribution(self) -> None:
        deterministic = CustomPacman(ghost_rand_prob=0.0)
        randomised = CustomPacman(ghost_rand_prob=0.8)
        try:
            state, _ = deterministic.reset(seed=1)
            successors_a, probs_a = deterministic._lazy_successor_distribution(state, 1)  # noqa: SLF001
            successors_b, probs_b = randomised._lazy_successor_distribution(state, 1)  # noqa: SLF001
            self.assertAlmostEqual(float(probs_a.sum()), 1.0)
            self.assertAlmostEqual(float(probs_b.sum()), 1.0)
            dist_a = dict(zip(successors_a, probs_a, strict=True))
            dist_b = dict(zip(successors_b, probs_b, strict=True))
            self.assertNotEqual(dist_a, dist_b)
        finally:
            deterministic.close()
            randomised.close()

    def test_render_ansi_and_rgb_array(self) -> None:
        grid = CustomColourGridWorld(render_mode="ansi")
        try:
            grid.reset(seed=1)
            self.assertIsInstance(grid.render(), str)
        finally:
            grid.close()

        pacman = CustomMiniPacman(render_mode="rgb_array")
        try:
            pacman.reset(seed=1)
            frame = pacman.render()
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[-1], 3)
        finally:
            pacman.close()

    def test_colour_bomb_uses_masa_renderer_and_marks_bombs(self) -> None:
        env = CustomColourBombGridWorld(render_mode="ansi")
        try:
            env.reset(seed=1)
            self.assertIsInstance(env._renderer, ColourBombGridWorldRenderer)  # noqa: SLF001
            self.assertIn("X", env.render())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
