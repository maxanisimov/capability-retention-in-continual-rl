"""Rendering preserves reset state and uses real rgb_array environment output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from projects.safe_policy_optimisation.scripts import (
    render_frozenlake_initial_frames as renderer,
)


class FrozenLakeInitialFramesTests(unittest.TestCase):
    def test_native_render_is_called_without_taking_an_action(self):
        with tempfile.TemporaryDirectory() as temp:
            layout = Path(temp) / "layout.txt"
            layout.write_text("\n".join(renderer.frozen.structured_layout(16)) + "\n")
            config = {
                "env_id": renderer.frozen.ENV_ID,
                "env_kwargs": {
                    "size": 16,
                    "is_slippery": True,
                    "success_rate": 0.8,
                    "step_penalty": 0.001,
                },
                "max_episode_steps": 625,
            }
            original_render = renderer.frozen.SparseFrozenLake.render
            captured = []

            def track_render(env):
                self.assertEqual(env.render_mode, "rgb_array")
                self.assertEqual(env.s, 0)
                self.assertIsNone(env.lastaction)
                frame = original_render(env)
                captured.append(frame)
                return frame

            with (
                patch.object(
                    renderer.frozen.SparseFrozenLake,
                    "step",
                    side_effect=AssertionError("No actions permitted"),
                ),
                patch.object(renderer.frozen.SparseFrozenLake, "render", track_render),
            ):
                frame = renderer.render_initial_frame(config, layout, tile_pixels=16)
            self.assertEqual(len(captured), 1)
            self.assertIs(frame, captured[0])
            self.assertEqual(frame.shape, (256, 256, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertGreater(len(np.unique(frame.reshape(-1, 3), axis=0)), 10)

    def test_invalid_tile_resolution_is_rejected_before_creating_environment(self):
        with self.assertRaises(ValueError):
            renderer.render_initial_frame({}, Path("unused"), tile_pixels=0)


if __name__ == "__main__":
    unittest.main()
