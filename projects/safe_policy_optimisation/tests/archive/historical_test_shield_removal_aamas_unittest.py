"""Preserve the paired shield-removal experiment and seed-level uncertainty."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from projects.safe_policy_optimisation.scripts import plot_shield_removal_comparison_aamas as plot


class ShieldRemovalAamasTests(unittest.TestCase):
    def test_shield_on_off_share_weights_but_use_different_evaluation_sections(self):
        on, off, pspo = [spec for spec, _color, _hatch in plot.SERIES]
        self.assertEqual(on.relative_path, off.relative_path)
        self.assertEqual(on.section, ("shielded",))
        self.assertEqual(off.section, ("nominal",))
        self.assertTrue(pspo.adaptive)
        self.assertEqual([plot.DISPLAY_LABELS[spec.key] for spec in (on, off, pspo)],
                         ["PPO-Shield (shield on)", "PPO-Shield (shield off)", "PSPO"])

    def test_only_shield_off_is_hatched_and_exact_legend_fits_one_row(self):
        methods = plot.bar_methods()
        self.assertEqual([method.hatch for method in methods], [None, "///", None])
        panel = plot.BarPanel("Example", [1, 0.5, 1], [0, 0.1, 0],
                              [1, 0.5, 1], [0, 0.1, 0])
        fig = plot.compact_figure([panel] * 6, methods, transpose=True, legend_one_row=True)
        try:
            for axis in fig.axes:
                self.assertEqual([bar.get_hatch() for bar in axis.patches], [None, "///", None])
            legend = fig.legends[0]
            self.assertEqual([patch.get_hatch() for patch in legend.get_patches()],
                             [None, "///", None])
            texts = legend.get_texts()
            self.assertEqual([text.get_text() for text in texts],
                             ["PPO-Shield (shield on)", "PPO-Shield (shield off)", "PSPO"])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            box = legend.get_window_extent(renderer)
            self.assertTrue(fig.bbox.contains(box.x0, box.y0))
            self.assertTrue(fig.bbox.contains(box.x1, box.y1))
            self.assertEqual(len({round(text.get_window_extent(renderer).y0, 3)
                                  for text in texts}), 1)
            self.assertTrue(all(text.get_fontsize() == 7 for text in texts))
        finally:
            plt.close(fig)

    def test_aggregate_is_mean_and_two_se_of_seed_values(self):
        rewards = list(range(10))
        safeties = [0.1 * value for value in rewards]
        with patch.object(plot, "read_seed_values", return_value=(rewards, safeties, {100})):
            panel = plot.load_panels(plot.ENVIRONMENTS[:1])[0]
        rm, re = plot.mean_and_error(rewards, 2)
        sm, se = plot.mean_and_error(safeties, 2)
        self.assertEqual(panel.reward_means, [rm] * 3)
        self.assertEqual(panel.reward_errors, [re] * 3)
        self.assertEqual(panel.safety_means, [sm] * 3)
        self.assertEqual(panel.safety_errors, [se] * 3)

    def test_incomplete_seeds_and_mixed_evaluation_protocols_are_rejected(self):
        for values in (([1] * 9, [1] * 9, {100}), ([1] * 10, [1] * 10, {20, 100})):
            with patch.object(plot, "read_seed_values", return_value=values):
                with self.assertRaises(ValueError):
                    plot.load_panels(plot.ENVIRONMENTS[:1])

    def test_invalid_error_multiplier_is_rejected(self):
        for multiplier in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                plot.load_panels(plot.ENVIRONMENTS[:1], multiplier)


if __name__ == "__main__":
    unittest.main()
