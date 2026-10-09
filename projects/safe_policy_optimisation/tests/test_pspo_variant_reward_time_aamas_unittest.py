"""Check seed-level uncertainty and unambiguous runtime-log parsing."""

import copy
import csv
import math
import re
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from projects.safe_policy_optimisation.scripts.plot_pspo_variant_reward_time_aamas import (
    ALL_VARIANTS,
    COLUMN_HEIGHT_IN,
    COLUMN_WIDTH_IN,
    ENVIRONMENTS,
    HEIGHT_IN,
    METRICS,
    VARIANTS,
    WIDTH_IN,
    build_figure,
    full_width_height,
    mean_two_se,
    plot,
    read_elapsed_logs,
    read_summary_csv,
    single_column_height,
    write_figure_snippet,
)


def example_rows(variants=VARIANTS):
    rewards = [-1.45, 1.0, 24.6, 1.0, 0.9, 0.956]
    minutes = [2.7, 1.7, 15.0, 10.2, 81.7, 538.5]
    rows = []
    for index, (environment, _) in enumerate(ENVIRONMENTS):
        for variant_index, (variant, label, _, _) in enumerate(variants):
            rows.append(
                {
                    "environment": environment,
                    "variant": variant,
                    "label": label,
                    "n_seeds": 10,
                    "reward_mean": rewards[index],
                    "reward_two_se": 0.2 if index == 4 else 0.05,
                    "rl_stage_minutes_mean": minutes[index] * (1 - 0.1 * variant_index),
                    "rl_stage_minutes_two_se": minutes[index] * 0.05,
                    "safety_rate_mean": 1.0,
                }
            )
    return rows


class VariantFigureStatisticsTests(unittest.TestCase):
    def test_requested_environment_order_and_variant_labels(self):
        self.assertEqual(
            [environment for environment, _ in ENVIRONMENTS],
            [
                "media_streaming",
                "colour_bomb",
                "colour_bomb_v2",
                "bridge_crossing",
                "bridge_crossing_v2",
                "mini_pacman",
            ],
        )
        self.assertEqual(
            [label for _, label, _, _ in VARIANTS],
            [
                "PSPO orthotope (default)",
                "PSPO orthotope verify-first",
                "PSPO line segment",
            ],
        )

    def test_two_se_uses_unbiased_sample_standard_deviation(self):
        mean, error = mean_two_se([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(mean, 2.5)
        self.assertAlmostEqual(error, math.sqrt(5 / 3))

    def test_identical_seed_means_have_zero_error(self):
        self.assertEqual(mean_two_se([1.0] * 10), (1.0, 0.0))

    def test_invalid_observations_are_rejected(self):
        for values in ([1.0], [1.0, float("nan")], [1.0, float("inf")]):
            with self.assertRaises(ValueError):
                mean_two_se(values)

    def test_runtime_logs_require_exactly_ten_distinct_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(
                "\n".join(f"ok seed{i} core=42 {100 + i}s" for i in range(10))
            )
            elapsed = read_elapsed_logs([path])
            self.assertEqual(elapsed[9], 109)
            with self.assertRaises(ValueError):
                read_elapsed_logs([path, path])
            path.write_text("ok seed0 core=42 100s\n")
            with self.assertRaises(ValueError):
                read_elapsed_logs([path])


class VariantFigureLayoutTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_native_sizes_and_metric_orientation(self):
        for single_column, size, geometry in (
            (False, (WIDTH_IN, HEIGHT_IN), (2, 6)),
            (True, (COLUMN_WIDTH_IN, COLUMN_HEIGHT_IN), (6, 2)),
        ):
            fig = build_figure(example_rows(), single_column=single_column)
            self.assertEqual(tuple(fig.get_size_inches()), size)
            self.assertEqual(len(fig.axes), 12)
            for axis in fig.axes:
                self.assertEqual(
                    axis.get_subplotspec().get_gridspec().get_geometry(), geometry
                )
                self.assertTrue(all(bar.get_hatch() is None for bar in axis.patches))
                self.assertTrue(
                    all(tick.get_fontsize() == 7 for tick in axis.get_yticklabels())
                )
            self.assertTrue(
                all(patch.get_hatch() is None for patch in fig.legends[0].get_patches())
            )

    def test_statistics_and_error_endpoints_are_unchanged(self):
        rows = example_rows()
        before = copy.deepcopy(rows)
        lookup = {(row["environment"], row["variant"]): row for row in rows}
        for single_column in (False, True):
            fig = build_figure(rows, single_column=single_column)
            for environment_index, (environment, _) in enumerate(ENVIRONMENTS):
                for metric_index, (mean_key, error_key) in enumerate(METRICS):
                    axis_index = (
                        2 * environment_index + metric_index
                        if single_column
                        else 6 * metric_index + environment_index
                    )
                    axis = fig.axes[axis_index]
                    for position, (variant, _, _, _) in enumerate(VARIANTS):
                        source = lookup[environment, variant]
                        bar = axis.patches[position]
                        self.assertEqual(bar.get_y(), 0)
                        self.assertAlmostEqual(bar.get_height(), source[mean_key])
                        interval = axis.collections[position].get_segments()[0]
                        low, high = (
                            source[mean_key] - source[error_key],
                            source[mean_key] + source[error_key],
                        )
                        self.assertAlmostEqual(interval[0][1], low)
                        self.assertAlmostEqual(interval[1][1], high)
                        self.assertLessEqual(axis.get_ylim()[0], low)
                        self.assertGreaterEqual(axis.get_ylim()[1], high)
        self.assertEqual(rows, before)

    def test_headings_legend_and_visible_ticks_do_not_overlap(self):
        for single_column in (False, True):
            fig = build_figure(example_rows(), single_column=single_column)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend = fig.legends[0]
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()],
                [label for _, label, _, _ in VARIANTS],
            )
            artists = [legend] + fig.texts
            if not single_column:
                artists += [axis.title for axis in fig.axes[:6]]
                self.assertEqual(
                    [axis.get_title() for axis in fig.axes[:6]],
                    [title for _, title in ENVIRONMENTS],
                )
            else:
                self.assertEqual(
                    [text.get_text() for text in fig.texts[:6]],
                    [title.replace("\n", " ") for _, title in ENVIRONMENTS],
                )
                self.assertTrue(
                    all(text.get_position()[0] == 0.5 for text in fig.texts[:6])
                )
            boxes = [artist.get_window_extent(renderer) for artist in artists]
            for index, box in enumerate(boxes):
                self.assertTrue(fig.bbox.contains(box.x0, box.y0))
                self.assertTrue(fig.bbox.contains(box.x1, box.y1))
                for other in boxes[index + 1 :]:
                    self.assertFalse(box.overlaps(other))
                for axis in fig.axes:
                    self.assertFalse(box.overlaps(axis.get_window_extent(renderer)))
                    for tick in axis.get_yticklabels():
                        if (
                            axis.get_ylim()[0]
                            <= tick.get_position()[1]
                            <= axis.get_ylim()[1]
                        ):
                            self.assertFalse(
                                box.overlaps(tick.get_window_extent(renderer))
                            )

    def test_all_variants_layout_keeps_panels_and_fits_its_legend(self):
        rows = example_rows(ALL_VARIANTS)
        for single_column, size in (
            (False, (WIDTH_IN, full_width_height(ALL_VARIANTS))),
            (True, (COLUMN_WIDTH_IN, single_column_height(ALL_VARIANTS))),
        ):
            fig = build_figure(rows, single_column=single_column, variants=ALL_VARIANTS)
            self.assertEqual(tuple(fig.get_size_inches()), size)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend = fig.legends[0]
            self.assertEqual(
                [text.get_text() for text in legend.get_texts()],
                [label for _, label, _, _ in ALL_VARIANTS],
            )
            box = legend.get_window_extent(renderer)
            self.assertTrue(fig.bbox.contains(box.x0, box.y0))
            self.assertTrue(fig.bbox.contains(box.x1, box.y1))
            for axis in fig.axes:
                self.assertEqual(len(axis.patches), len(ALL_VARIANTS))
                self.assertFalse(box.overlaps(axis.get_window_extent(renderer)))
                self.assertFalse(box.overlaps(axis.title.get_window_extent(renderer)))
            for text in fig.texts:
                self.assertFalse(box.overlaps(text.get_window_extent(renderer)))

    def test_missing_duplicate_or_invalid_rows_are_rejected(self):
        rows = example_rows()
        cases = [rows[:-1], rows + [rows[0]]]
        for key, value in (
            ("n_seeds", 9),
            ("reward_mean", float("nan")),
            ("reward_two_se", -1.0),
            ("rl_stage_minutes_two_se", float("inf")),
        ):
            modified = copy.deepcopy(rows)
            modified[0][key] = value
            cases.append(modified)
        for case in cases:
            with self.assertRaises(ValueError):
                build_figure(case)

    def test_export_has_native_pdf_dimensions_fonts_and_matching_snippets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "summary.csv"
            rows = example_rows()
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            original = source.read_bytes()
            rows = read_summary_csv(source)
            for single_column, size in (
                (False, (WIDTH_IN, HEIGHT_IN)),
                (True, (COLUMN_WIDTH_IN, COLUMN_HEIGHT_IN)),
            ):
                suffix = "_column" if single_column else ""
                stem = output / f"pspo_variant_reward_time_aamas{suffix}"
                plot(rows, stem, single_column=single_column)
                write_figure_snippet(output, single_column=single_column)
                pdf = stem.with_suffix(".pdf").read_bytes()
                box = re.search(rb"/MediaBox\s*\[([^\]]+)\]", pdf).group(1)
                dimensions = [float(value) / 72 for value in box.split()][2:]
                for actual, expected in zip(dimensions, size):
                    self.assertAlmostEqual(actual, expected)
                self.assertIn(b"/FontFile2", pdf)
                self.assertNotIn(b"/Subtype /Type3", pdf)
                snippet = (output / f"figure_aamas{suffix}.tex").read_text()
                environment = "figure" if single_column else "figure*"
                width = r"\columnwidth" if single_column else r"\textwidth"
                self.assertIn(rf"\begin{{{environment}}}[t]", snippet)
                self.assertIn(f"width={width}", snippet)
                self.assertIn(r"\Description{", snippet)
                self.assertIn(r"\label{fig:pspo-variants-reward-time}", snippet)
                self.assertNotIn("\b", snippet)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
