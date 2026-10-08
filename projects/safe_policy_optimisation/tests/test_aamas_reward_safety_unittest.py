"""Native AAMAS dimensions, honest uncertainty, and compact legend layout."""

import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from projects.safe_policy_optimisation.scripts._aamas_reward_safety import (
    BarMethod, BarPanel, COLUMN_WIDTH_IN, ENVIRONMENT_ORDER, FULL_WIDTH_IN,
    GRID_HEIGHT_IN, SINGLE_HEIGHT_IN, compact_figure, legend_indices,
    ordered_panels, save_compact_figure, TRANSPOSED_SINGLE_HEIGHT_IN,
)


class AamasRewardSafetyTests(unittest.TestCase):
    def setUp(self):
        self.methods = [BarMethod("a", "Method A", "green"),
                        BarMethod("b", "Method B", "blue")]
        self.panel = BarPanel("Example", [-5.0, 2.0], [0.7, 0.2],
                              [0.95, 0.2], [0.1, 0.3])

    def tearDown(self):
        plt.close("all")

    def test_full_width_and_single_column_native_dimensions(self):
        fig = compact_figure([self.panel] * 6, self.methods)
        self.assertEqual(tuple(fig.get_size_inches()), (FULL_WIDTH_IN, GRID_HEIGHT_IN))
        self.assertEqual(len(fig.axes), 12)
        fig = compact_figure([self.panel], self.methods, single_column=True)
        self.assertEqual(tuple(fig.get_size_inches()), (COLUMN_WIDTH_IN, SINGLE_HEIGHT_IN))
        self.assertEqual(len(fig.axes), 2)

    def test_requested_environment_order_and_canonical_names(self):
        labels = ["MiniPacman", "Bridge Crossing v2", "Bridge Crossing",
                  "Colour Bomb v2", "Colour Bomb", "Media Streaming"]
        panels = [BarPanel(label, [1], [0], [1], [0]) for label in labels]
        from projects.safe_policy_optimisation.scripts._aamas_reward_safety import canonical_label
        self.assertEqual(tuple(canonical_label(p.label) for p in ordered_panels(panels)),
                         ENVIRONMENT_ORDER)

    def test_transposed_grid_is_single_column_with_metrics_across_columns(self):
        panels = [BarPanel(label, [-5, 2], [0.7, 0.2], [0.95, 0.2], [0.1, 0.3])
                  for label in reversed(ENVIRONMENT_ORDER)]
        fig = compact_figure(panels, self.methods, transpose=True)
        self.assertAlmostEqual(fig.get_size_inches()[0], COLUMN_WIDTH_IN)
        self.assertAlmostEqual(fig.get_size_inches()[1], 3.70)
        self.assertEqual(len(fig.axes), 12)
        for row, label in enumerate(ENVIRONMENT_ORDER):
            reward, safety = fig.axes[2 * row:2 * row + 2]
            heading = next(text for text in fig.texts if text.get_text() == label)
            self.assertEqual(heading.get_ha(), "center")
            self.assertEqual(heading.get_position()[0], 0.57)
            self.assertGreater(heading.get_position()[1], reward.get_position().y1)
            self.assertEqual([round(p.get_y() + p.get_height(), 10)
                              for p in reward.patches], [-5, 2])
            self.assertEqual([round(p.get_y() + p.get_height(), 10)
                              for p in safety.patches], [0.95, 0.2])
        self.assert_legend_fits(fig)

    def test_solid_bars_and_legend_in_every_layout(self):
        for kwargs in ({}, {"single_column": True}, {"transpose": True},
                       {"single_column": True, "transpose": True}):
            fig = compact_figure([self.panel], self.methods, **kwargs)
            for axis in fig.axes:
                self.assertTrue(all(patch.get_hatch() is None for patch in axis.patches))
            self.assertTrue(all(patch.get_hatch() is None
                                for patch in fig.legends[0].get_patches()))

    def test_transposed_metric_columns_have_a_narrow_gap(self):
        fig = compact_figure([self.panel] * 6, self.methods, transpose=True)
        reward, safety = fig.axes[:2]
        gap_inches = (safety.get_position().x0 - reward.get_position().x1) * COLUMN_WIDTH_IN
        self.assertLess(gap_inches, 0.30)

    def test_shifted_row_headings_clear_neighbouring_axes_and_tick_labels(self):
        panels = [BarPanel(label, [-5, 2], [0.7, 0.2], [0.95, 0.2], [0.1, 0.3])
                  for label in ENVIRONMENT_ORDER]
        fig = compact_figure(panels, self.methods, transpose=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label in ENVIRONMENT_ORDER:
            heading = next(text for text in fig.texts if text.get_text() == label)
            box = heading.get_window_extent(renderer)
            for axis in fig.axes:
                self.assertFalse(box.overlaps(axis.get_window_extent()))
                for tick in axis.get_yticklabels():
                    if axis.get_ylim()[0] <= tick.get_position()[1] <= axis.get_ylim()[1]:
                        self.assertFalse(box.overlaps(tick.get_window_extent(renderer)))

    def assert_legend_fits(self, fig):
        fig.canvas.draw()
        legend = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
        self.assertTrue(fig.bbox.contains(legend.x0, legend.y0))
        self.assertTrue(fig.bbox.contains(legend.x1, legend.y1))
        for axis in fig.axes:
            self.assertFalse(legend.overlaps(axis.get_window_extent()))

    def test_transposed_single_task_has_stacked_metrics_and_side_legend(self):
        methods = [BarMethod("a", "PPO-Shield\nshield on", "green"),
                   BarMethod("b", "PPO-Shield\nshield off", "blue")]
        fig = compact_figure([self.panel], methods, single_column=True, transpose=True)
        self.assertEqual(tuple(fig.get_size_inches()),
                         (COLUMN_WIDTH_IN, TRANSPOSED_SINGLE_HEIGHT_IN))
        self.assertEqual([axis.get_title() for axis in fig.axes],
                         ["Total reward", "Safety rate"])
        self.assertEqual(fig._suptitle.get_position()[0], 0.57)
        self.assertGreater(fig.axes[0].get_position().y0, fig.axes[1].get_position().y1)
        self.assert_legend_fits(fig)

    def test_transposed_seven_method_legend_fits(self):
        labels = ["PPO", "PPO-Lagrangian", "PPO-PID-Lagrangian", "CPO",
                  "PPO-Shield", "PPO-Shield-Nominal", "PSPO"]
        methods = [BarMethod(str(i), label, "green") for i, label in enumerate(labels)]
        panel = BarPanel("Example", [1] * 7, [0.1] * 7, [1] * 7, [0.1] * 7)
        fig = compact_figure([panel] * 6, methods, transpose=True)
        self.assert_legend_fits(fig)

    def test_shield_off_bar_sits_between_ppo_shield_and_pspo(self):
        from projects.safe_policy_optimisation.scripts.plot_final_policy_reward_safety_bars import (
            bar_methods,
        )
        self.assertEqual([spec.key for spec in bar_methods()][-2:], ["ppo_shield", "pspo"])
        specs = bar_methods(include_shield_off=True)
        self.assertEqual([spec.key for spec in specs][-3:],
                         ["ppo_shield", "ppo_shield_nominal", "pspo"])
        self.assertEqual(specs[-2].label, "PPO Shield (shield off)")
        methods = [BarMethod(spec.key, spec.label, "green") for spec in specs]
        panel = BarPanel("Example", [1] * 7, [0.1] * 7, [1] * 7, [0.1] * 7)
        fig = compact_figure([panel] * 6, methods)
        self.assertEqual(tuple(fig.get_size_inches()), (FULL_WIDTH_IN, GRID_HEIGHT_IN))
        self.assert_legend_fits(fig)

    def test_shortened_one_row_legend_fits_without_shrinking_text(self):
        labels = ["PPO", "Lagrangian", "PID-Lagrangian", "CPO", "Shield", "PSPO"]
        methods = [BarMethod(str(i), label, "green") for i, label in enumerate(labels)]
        panel = BarPanel("Example", [1] * 6, [0.1] * 6, [1] * 6, [0.1] * 6)
        fig = compact_figure([panel] * 6, methods, transpose=True, legend_one_row=True)
        self.assertAlmostEqual(fig.get_size_inches()[0], 3.33)
        self.assertAlmostEqual(fig.get_size_inches()[1], 3.44)
        self.assert_legend_fits(fig)
        texts = fig.legends[0].get_texts()
        self.assertTrue(all(text.get_fontsize() == 7 for text in texts))
        renderer = fig.canvas.get_renderer()
        self.assertEqual(len({round(text.get_window_extent(renderer).y0, 3)
                              for text in texts}), 1)

    def test_uncertainty_is_not_clipped_at_probability_bounds(self):
        fig = compact_figure([self.panel], self.methods)
        self.assertLessEqual(fig.axes[0].get_ylim()[0], -5.7)
        self.assertGreaterEqual(fig.axes[0].get_ylim()[1], 2.2)
        self.assertLessEqual(fig.axes[1].get_ylim()[0], -0.1 + 1e-12)
        self.assertGreaterEqual(fig.axes[1].get_ylim()[1], 1.05)
        for axis, means in zip(fig.axes, (self.panel.reward_means, self.panel.safety_means)):
            self.assertEqual([round(bar.get_y() + bar.get_height(), 10)
                              for bar in axis.patches], means)

    def test_legend_reads_in_bar_order(self):
        self.assertEqual(legend_indices(7, 4), [0, 4, 1, 5, 2, 6, 3])
        self.assertEqual(legend_indices(6, 3), [0, 3, 1, 4, 2, 5])

    def test_missing_results_are_not_zero_bars(self):
        panel = BarPanel("Example", [1, None], [0.1, 0], [1, None], [0, 0])
        fig = compact_figure([panel], self.methods)
        for axis in fig.axes:
            self.assertEqual(len(axis.patches), 1)
            self.assertIn("n/a", [text.get_text() for text in axis.texts])

    def test_bad_inputs_fail_explicitly(self):
        for panel in (BarPanel("Empty", [None, None], [0, 0], [1, 1], [0, 0]),
                      BarPanel("Empty", [1, 1], [0, 0], [None, None], [0, 0]),
                      BarPanel("Negative", [1, 1], [-0.1, 0], [1, 1], [0, 0]),
                      BarPanel("Invalid", [float("nan"), 1], [0, 0], [1, 1], [0, 0])):
            with self.assertRaises(ValueError):
                compact_figure([panel], self.methods)

    def test_export_preserves_errors_and_provides_accessible_latex(self):
        for multiplier, single in ((1.0, False), (2.0, True)):
            with tempfile.TemporaryDirectory() as directory:
                stem = Path(directory) / "example"
                fig = compact_figure([self.panel], self.methods, single_column=single)
                save_compact_figure(fig, stem, panels=[self.panel], methods=self.methods,
                                    se_multiplier=multiplier, caption="Example.")
                with stem.with_suffix(".csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(float(rows[0]["reward_error"]), 0.7)
                self.assertEqual(float(rows[0]["se_multiplier"]), multiplier)
                snippet = stem.with_suffix(".tex").read_text()
                self.assertIn("\\Description{", snippet)
                self.assertIn("\\begin{figure}" if single else "\\begin{figure*}", snippet)
                self.assertIn("minimum", snippet)
                self.assertIn("solid-colour", snippet)
                self.assertNotIn("hatch pattern", snippet)
                self.assertTrue(stem.with_suffix(".pdf").is_file())
                self.assertTrue(stem.with_suffix(".png").is_file())


if __name__ == "__main__":
    unittest.main()
