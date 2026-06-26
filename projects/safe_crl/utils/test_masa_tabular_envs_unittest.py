"""Unit tests for custom MASA-style tabular environments."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import gymnasium as gym

    import projects.safe_crl.utils.masa_tabular_envs  # noqa: F401
    from projects.safe_crl.utils.masa_tabular_envs.factory import make_custom_masa_env
    from projects.safe_crl.utils.masa_tabular_envs.frozen_lake import CustomFrozenLake
    from projects.safe_crl.utils.masa_tabular_envs.gridworlds import CustomColourBombGridWorld, CustomColourGridWorld
    from projects.safe_crl.utils.masa_tabular_envs.media_streaming import CustomMediaStreaming
    from projects.safe_crl.utils.masa_tabular_envs.pacman import CustomMiniPacman, CustomPacman
    from projects.safe_crl.utils.masa_tabular_envs.renderers.colour_bomb_grid_world import ColourBombGridWorldRenderer
except ModuleNotFoundError:  # pragma: no cover - exercised only without RL extras
    gym = None


@unittest.skipIf(gym is None, "gymnasium is not installed")
class CustomMasaTabularEnvTests(unittest.TestCase):
    ENV_IDS = [
        "CustomFrozenLake-v0",
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

    def test_factory_creates_custom_frozen_lake(self) -> None:
        env = make_custom_masa_env(
            "CustomFrozenLake-v0",
            max_episode_steps=9,
            env_kwargs={"desc": ["SF", "HG"], "is_slippery": False},
        )
        try:
            obs, info = env.reset(seed=1)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(env.spec.max_episode_steps, 9)
            self.assertEqual(env.unwrapped.nrow, 2)
            self.assertEqual(env.unwrapped.ncol, 2)
        finally:
            env.close()

    def test_frozen_lake_matches_gymnasium_transition_model(self) -> None:
        cases = [
            {"desc": ["SF", "HG"], "is_slippery": False},
            {"desc": ["SFF", "FHF", "FFG"], "is_slippery": True},
            {"desc": ["SFF", "FHF", "FFG"], "is_slippery": True, "success_rate": 0.7},
            {"desc": ["SF", "HG"], "is_slippery": False, "reward_schedule": (3, -2, -0.5)},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                custom = CustomFrozenLake(**kwargs)
                reference = gym.make("FrozenLake-v1", **kwargs)
                try:
                    self.assertEqual(custom.P, reference.unwrapped.P)
                    matrix = custom.get_transition_matrix()
                    self.assertEqual(matrix.shape, (custom.nrow * custom.ncol, custom.nrow * custom.ncol, 4))
                    self.assertTrue(np.allclose(matrix.sum(axis=0), 1.0))
                    successors = custom.get_successor_states_dict()
                    self.assertIsNotNone(successors)
                finally:
                    custom.close()
                    reference.close()

    def test_frozen_lake_labels_and_costs(self) -> None:
        env = CustomFrozenLake(desc=["SF", "HG"], is_slippery=False)
        try:
            self.assertEqual(env.label_fn(0), {"start"})
            self.assertEqual(env.label_fn(1), {"frozen"})
            self.assertEqual(env.label_fn(2), {"hole"})
            self.assertEqual(env.label_fn(3), {"goal"})
            self.assertEqual(env.cost_fn(env.label_fn(2)), 1.0)
            self.assertEqual(env.cost_fn(env.label_fn(3)), 0.0)
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

    def test_frozen_lake_render_matches_gymnasium(self) -> None:
        kwargs = {"desc": ["SF", "HG"], "is_slippery": False}
        custom_ansi = CustomFrozenLake(render_mode="ansi", **kwargs)
        reference_ansi = gym.make("FrozenLake-v1", render_mode="ansi", **kwargs)
        try:
            custom_ansi.reset(seed=1)
            reference_ansi.reset(seed=1)
            self.assertEqual(custom_ansi.render(), reference_ansi.render())
        finally:
            custom_ansi.close()
            reference_ansi.close()

        try:
            import pygame  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("pygame is not installed")

        custom_rgb = CustomFrozenLake(render_mode="rgb_array", **kwargs)
        reference_rgb = gym.make("FrozenLake-v1", render_mode="rgb_array", **kwargs)
        try:
            custom_rgb.reset(seed=1)
            reference_rgb.reset(seed=1)
            self.assertTrue(np.array_equal(custom_rgb.render(), reference_rgb.render()))
        finally:
            custom_rgb.close()
            reference_rgb.close()

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
