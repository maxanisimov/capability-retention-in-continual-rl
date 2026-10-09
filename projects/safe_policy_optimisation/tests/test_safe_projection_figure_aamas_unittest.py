"""Single-column projection comparison preserves rewards and uncertainty."""

import copy
import csv
import re
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from projects.safe_policy_optimisation.scripts import generate_safe_projection_comparison_assets as plot


def example_rows():
    rows = []
    for environment in plot.ENVIRONMENTS:
        for index, method in enumerate(plot.METHODS):
            rows.append({"environment": environment, "method": method, "n_seeds": 10,
                         "mean_total_reward": -1.5 + index * 0.02 if environment == "media_streaming"
                         else 0.5 + index * 0.08,
                         "reward_2se": 0.05, "mean_safety_rate": 1.0, "safety_2se": 0.0})
    return rows


class SafeProjectionFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_native_single_column_geometry_and_environment_order(self):
        fig = plot.build_reward_figure(example_rows())
        self.assertEqual(tuple(fig.get_size_inches()), (3.33, 2.98))
        self.assertEqual(len(fig.axes), 6)
        self.assertEqual([axis.get_title().replace("\n", " ") for axis in fig.axes],
                         list(plot.ENVIRONMENTS.values()))
        for axis in fig.axes:
            self.assertEqual(axis.get_subplotspec().get_gridspec().get_geometry(), (3, 2))
            self.assertEqual(axis.title.get_fontsize(), 8)
            filled = axis.patches[:len(plot.PLOT_METHODS)]
            self.assertEqual([bar.get_hatch() for bar in filled],
                             [None if method == "pspo" else plot.PROJECTED_HATCH
                              for method in plot.PLOT_METHODS])

    def test_bar_tops_and_error_endpoints_are_unchanged(self):
        rows = example_rows()
        lookup = {(row["environment"], row["method"]): row for row in rows}
        fig = plot.build_reward_figure(rows)
        for axis, environment in zip(fig.axes, plot.ENVIRONMENTS):
            for index, (bar, method) in enumerate(zip(axis.patches, plot.PLOT_METHODS)):
                source = lookup[environment, method]
                self.assertAlmostEqual(bar.get_y() + bar.get_height(), source["mean_total_reward"])
                segments = axis.collections[0].get_segments()
                self.assertAlmostEqual(segments[index][0][1],
                                       source["mean_total_reward"] - source["reward_2se"])
                self.assertAlmostEqual(segments[index][1][1],
                                       source["mean_total_reward"] + source["reward_2se"])
                self.assertLessEqual(axis.get_ylim()[0], segments[index][0][1])
                self.assertGreaterEqual(axis.get_ylim()[1], segments[index][1][1])
        self.assertLess(fig.axes[0].patches[0].get_y(), -1.55)
        self.assertAlmostEqual(fig.axes[1].patches[0].get_y(), 0.375)

    def test_legend_and_headings_fit_inside_the_canvas(self):
        fig = plot.build_reward_figure(example_rows())
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend = fig.legends[0]
        self.assertTrue(all(text.get_fontsize() == 6.5 for text in legend.get_texts()))
        for artist in [legend] + [axis.title for axis in fig.axes]:
            box = artist.get_window_extent(renderer)
            self.assertTrue(fig.bbox.contains(box.x0, box.y0))
            self.assertTrue(fig.bbox.contains(box.x1, box.y1))
        legend_box = legend.get_window_extent(renderer)
        self.assertFalse(any(legend_box.overlaps(axis.get_window_extent()) for axis in fig.axes))

    def test_missing_duplicate_nonfinite_and_inconsistent_summaries_are_rejected(self):
        rows = example_rows()
        cases = [rows[:-1], rows + [rows[0]]]
        for field, value in (("n_seeds", 9), ("reward_2se", -1),
                             ("mean_total_reward", float("nan")), ("mean_safety_rate", 0.9)):
            modified = copy.deepcopy(rows)
            modified[0][field] = value
            cases.append(modified)
        for rows in cases:
            with self.assertRaises(ValueError):
                plot.build_reward_figure(rows)

    def test_export_uses_embedded_fonts_and_a_single_column_snippet(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = example_rows()
            csv_path = output / "reward_comparison.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            before = csv_path.read_bytes()
            plot.plot(output, plot.read_summary_csv(csv_path))
            plot.write_figure_snippet(output)
            self.assertEqual(csv_path.read_bytes(), before)
            pdf = (output / (plot.STEM + ".pdf")).read_bytes()
            box = re.search(rb"/MediaBox\s*\[([^\]]+)\]", pdf).group(1)
            dims = [float(value) / 72 for value in box.split()][2:]
            self.assertAlmostEqual(dims[0], 3.33)
            self.assertAlmostEqual(dims[1], 2.98)
            self.assertIn(b"/FontFile2", pdf)
            self.assertNotIn(b"/Subtype /Type3", pdf)
            snippet = (output / "reward_figure.tex").read_text()
            self.assertIn("\\begin{figure}[t]", snippet)
            self.assertIn("width=3.33in", snippet)
            self.assertIn("\\Description{", snippet)
            self.assertIn("conditional projection", snippet)
            self.assertNotIn("\b", snippet)


if __name__ == "__main__":
    unittest.main()
