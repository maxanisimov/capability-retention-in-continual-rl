"""Unit tests for FrozenLake shield-safety initial-frame plotting helpers."""

from __future__ import annotations

import unittest

from experiments.pipelines.frozenlake_shield_safety.core.analysis.plot_initial_frames import (
    build_parser,
    default_initial_frame_dir,
    source_downstream_task_cfgs,
)


class FrozenLakeShieldSafetyInitialFramePlotTests(unittest.TestCase):
    def test_source_downstream_task_cfgs_use_pipeline_maps(self) -> None:
        source_cfg, downstream_cfg = source_downstream_task_cfgs("diagonal_10x10")

        self.assertEqual(source_cfg["env_map"][0], "SFHHHHHHHH")
        self.assertEqual(downstream_cfg["env_map"][0], "SFFHHHHHHH")
        self.assertEqual(source_cfg["max_episode_steps"], 100)
        self.assertEqual(downstream_cfg["max_episode_steps"], 100)
        self.assertFalse(source_cfg["is_slippery"])
        self.assertFalse(downstream_cfg["is_slippery"])

    def test_parser_defaults_to_dedicated_initial_frame_directory(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.layout, "diagonal_4x4")
        self.assertEqual(args.output_dir, default_initial_frame_dir())
        self.assertEqual(args.formats, ["pdf", "png"])


if __name__ == "__main__":
    unittest.main()
