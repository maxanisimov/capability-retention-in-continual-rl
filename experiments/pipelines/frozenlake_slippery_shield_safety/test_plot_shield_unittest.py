"""Unit tests for slippery FrozenLake shield probability plotting."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.pipelines.frozenlake_slippery_shield_safety.core.analysis.plot_shield import (
    _safe_probability,
    plot_shield_safety_probabilities,
)
from experiments.pipelines.frozenlake_slippery_shield_safety.core.analysis.plot_synthesised_shield import (
    build_parser,
    default_shield_figure_dir,
    save_synthesised_shield_plot,
)
from experiments.pipelines.frozenlake_slippery_shield_safety.core.config import SOURCE_MAP
from experiments.pipelines.frozenlake_slippery_shield_safety.core.env import make_env


class FrozenLakeSlipperyShieldPlotTests(unittest.TestCase):
    def test_safe_probability_is_action_risk_complement(self) -> None:
        action_risk = np.array([[0.2, 0.0, 1.0, -0.5]], dtype=np.float64)

        self.assertAlmostEqual(_safe_probability(action_risk, 0, 0), 0.8)
        self.assertAlmostEqual(_safe_probability(action_risk, 0, 1), 1.0)
        self.assertAlmostEqual(_safe_probability(action_risk, 0, 2), 0.0)
        self.assertAlmostEqual(_safe_probability(action_risk, 0, 3), 1.0)

    def test_plot_saves_shield_probability_overlay(self) -> None:
        shield = np.zeros((16, 4), dtype=np.int64)
        shield[0, 2] = 1
        action_risk = np.ones((16, 4), dtype=np.float64)
        action_risk[0, 2] = 0.12345
        env = make_env(
            SOURCE_MAP,
            task_num=0.0,
            max_episode_steps=16,
            is_slippery=True,
            success_rate=0.8,
            render_mode="rgb_array",
        )
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = Path(tmp_dir) / "shield.png"
                fig = plot_shield_safety_probabilities(
                    env,
                    shield,
                    action_risk=action_risk,
                    save_path=output_path,
                )
                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)
                self.assertIn("Shield safety probabilities", fig.axes[0].get_title())
                self.assertIn("0.877", [text.get_text() for text in fig.axes[0].texts])
        finally:
            env.close()
            if "fig" in locals():
                import matplotlib.pyplot as plt

                plt.close(fig)

    def test_synthesised_shield_script_saves_plot_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = save_synthesised_shield_plot(
                layout="diagonal_4x4",
                task="source",
                output_dir=Path(tmp_dir),
                basename="shield_demo",
                formats=["png"],
                success_rate=0.8,
                shield_type="probabilistic",
                shield_risk_threshold=0.05,
                save_tensors=False,
            )

            figure_path = Path(tmp_dir) / "shield_demo.png"
            metadata_path = Path(tmp_dir) / "shield_demo_metadata.yaml"
            self.assertEqual(result["figure_paths"], [figure_path])
            self.assertTrue(figure_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertGreater(figure_path.stat().st_size, 0)
            self.assertEqual(result["metadata"]["shield_type"], "probabilistic")
            self.assertEqual(
                result["metadata"]["probability_label"],
                "1 - action_risk = probability of eventually staying safe",
            )

    def test_synthesised_shield_parser_defaults_to_dedicated_directory(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.layout, "diagonal_4x4")
        self.assertEqual(args.task, "source")
        self.assertEqual(args.output_dir, default_shield_figure_dir())


if __name__ == "__main__":
    unittest.main()
