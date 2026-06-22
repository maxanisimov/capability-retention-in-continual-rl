"""Unit tests for custom MASA tabular shield visualisation helpers."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - exercised only without RL extras
    gym = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - exercised only without viz extras
    plt = None

try:
    import pygame  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised only without viz extras
    pygame = None

if gym is not None:
    import experiments.utils.masa_tabular_envs  # noqa: F401
    from experiments.utils.masa_tabular_envs.visualisation import (
        plot_tabular_shield,
        print_allowed_actions,
        render_tabular_shield_background,
    )


@unittest.skipIf(
    gym is None or plt is None or pygame is None,
    "gymnasium, matplotlib, and pygame are required",
)
class CustomMasaTabularVisualisationTests(unittest.TestCase):
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

    def test_plot_returns_axes_for_each_registered_custom_env(self) -> None:
        for env_id in self.ENV_IDS:
            with self.subTest(env_id=env_id):
                env = gym.make(env_id)
                try:
                    unwrapped = env.unwrapped
                    shield = _empty_shield(unwrapped)
                    shield[int(unwrapped._start_state), 0] = 1  # noqa: SLF001

                    background = render_tabular_shield_background(env)
                    self.assertIsInstance(background, np.ndarray)
                    self.assertEqual(background.ndim, 3)
                    self.assertEqual(background.shape[-1], 3)

                    ax = plot_tabular_shield(env, shield)
                    self.assertIsNotNone(ax)
                    plt.close(ax.figure)
                finally:
                    env.close()

    def test_background_matches_native_rgb_array_frame(self) -> None:
        env_ids = [
            "CustomFrozenLake-v0",
            "CustomColourGridWorld-v0",
            "CustomMiniPacman-v0",
            "CustomMediaStreaming-v0",
        ]
        for env_id in env_ids:
            with self.subTest(env_id=env_id):
                env = gym.make(env_id)
                try:
                    background = render_tabular_shield_background(env)
                    expected = _native_rgb_array(env.unwrapped)
                    np.testing.assert_array_equal(background, expected)
                finally:
                    env.close()

    def test_background_restores_render_mode(self) -> None:
        env = gym.make("CustomColourGridWorld-v0", render_mode="ansi")
        try:
            render_tabular_shield_background(env)
            self.assertEqual(env.unwrapped.render_mode, "ansi")
        finally:
            env.close()

    def test_plot_does_not_mutate_environment_state(self) -> None:
        env = gym.make("CustomMediaStreamingV3-v0")
        try:
            unwrapped = env.unwrapped
            unwrapped.reset(seed=7)
            unwrapped.step(1)
            before = {
                "_state": unwrapped._state,  # noqa: SLF001
                "_buffer_level": unwrapped._buffer_level,  # noqa: SLF001
                "_step_count": unwrapped._step_count,  # noqa: SLF001
                "_last_action": unwrapped._last_action,  # noqa: SLF001
                "render_mode": unwrapped.render_mode,
            }

            shield = _empty_shield(unwrapped)
            ax = plot_tabular_shield(env, shield)
            plt.close(ax.figure)

            after = {
                "_state": unwrapped._state,  # noqa: SLF001
                "_buffer_level": unwrapped._buffer_level,  # noqa: SLF001
                "_step_count": unwrapped._step_count,  # noqa: SLF001
                "_last_action": unwrapped._last_action,  # noqa: SLF001
                "render_mode": unwrapped.render_mode,
            }
            self.assertEqual(after, before)
        finally:
            env.close()

    def test_colour_bomb_v3_fixed_zone_maps_position_to_state(self) -> None:
        env = gym.make("CustomColourBombGridWorldV3-v0")
        try:
            unwrapped = env.unwrapped
            grid_area = int(unwrapped._grid_size**2)  # noqa: SLF001
            zone = 2
            base_state = 16
            state = zone * grid_area + base_state
            shield = _empty_shield(unwrapped)
            shield[state, 4] = 1

            output = _capture_print(
                env,
                shield,
                fixed_features={"zone": zone},
            )

            self.assertIn(f"state {state:4d}", output)
            self.assertIn("zone=2", output)
            self.assertIn("active_colour=red", output)

            ax = plot_tabular_shield(
                env,
                shield,
                fixed_features={"active_colour": "red"},
            )
            plt.close(ax.figure)
        finally:
            env.close()

    def test_pacman_slice_skips_walls_and_rejects_wall_ghost(self) -> None:
        env = gym.make("CustomMiniPacman-v0")
        try:
            shield = _empty_shield(env.unwrapped)
            shield[int(env.unwrapped._start_state), 4] = 1  # noqa: SLF001

            ax = plot_tabular_shield(env, shield)
            plt.close(ax.figure)

            with self.assertRaisesRegex(ValueError, "free Pacman cell"):
                plot_tabular_shield(
                    env,
                    shield,
                    fixed_features={"ghost_y": 0, "ghost_x": 0},
                )
        finally:
            env.close()

    def test_media_streaming_fixed_slices_are_printed(self) -> None:
        env_v2 = gym.make("CustomMediaStreamingV2-v0")
        try:
            state = int(env_v2.unwrapped._encode_state(3, 2))  # noqa: SLF001
            shield = _empty_shield(env_v2.unwrapped)
            shield[state, 1] = 1

            output = _capture_print(
                env_v2,
                shield,
                fixed_features={"fast_count": 2},
            )

            self.assertIn(f"state {state:4d}", output)
            self.assertIn("fast_count=2", output)
        finally:
            env_v2.close()

        env_v3 = gym.make("CustomMediaStreamingV3-v0")
        try:
            state = int(env_v3.unwrapped._encode_safety_state(2, 3))  # noqa: SLF001
            shield = _empty_shield(env_v3.unwrapped)
            shield[state, 0] = 1

            output = _capture_print(
                env_v3,
                shield,
                fixed_features={"time": 3},
            )

            self.assertIn(f"state {state:4d}", output)
            self.assertIn("time=3", output)
        finally:
            env_v3.close()

    def test_invalid_shape_and_fixed_features_raise_clear_errors(self) -> None:
        env = gym.make("CustomColourGridWorld-v0")
        try:
            unwrapped = env.unwrapped
            n_states = int(unwrapped._n_states)  # noqa: SLF001
            n_actions = int(unwrapped._n_actions)  # noqa: SLF001
            bad_shield = np.zeros(
                (n_states + 1, n_actions),
                dtype=int,
            )
            with self.assertRaisesRegex(ValueError, "shield must have shape"):
                plot_tabular_shield(env, bad_shield)

            shield = _empty_shield(unwrapped)
            with self.assertRaisesRegex(ValueError, "fixed_features"):
                plot_tabular_shield(env, shield, fixed_features={"zone": 0})
        finally:
            env.close()

        env_v3 = gym.make("CustomColourBombGridWorldV3-v0")
        try:
            shield = _empty_shield(env_v3.unwrapped)
            with self.assertRaisesRegex(ValueError, "zone must be"):
                plot_tabular_shield(env_v3, shield, fixed_features={"zone": 999})
        finally:
            env_v3.close()


def _empty_shield(env) -> np.ndarray:
    n_states = int(env._n_states)  # noqa: SLF001
    n_actions = int(env._n_actions)  # noqa: SLF001
    return np.zeros((n_states, n_actions), dtype=int)


def _native_rgb_array(env) -> np.ndarray:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    old_render_mode = getattr(unwrapped, "render_mode", None)
    try:
        unwrapped.render_mode = "rgb_array"
        frame = env.render()
    finally:
        unwrapped.render_mode = old_render_mode
    if frame is None:
        raise AssertionError("native rgb_array render returned None")
    return np.array(frame, copy=True)


def _capture_print(env, shield: np.ndarray, *, fixed_features: dict[str, int]) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_allowed_actions(env, shield, fixed_features=fixed_features)
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
