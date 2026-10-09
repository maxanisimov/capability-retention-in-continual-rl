"""Final figure styling, dimensions, metrics and 200k-step endpoint semantics."""

import unittest

from projects.safe_policy_optimisation.scripts import (
    plot_frozenlake_scalability as plot,
)


class ScalabilityFigureTests(unittest.TestCase):
    def setUp(self):
        rows = [
            dict(
                size=size,
                method=method,
                channel=channel,
                timestep=step,
                seed_count=10,
                reward_mean=0.2,
                reward_two_se=0.02,
                safety_mean=0.9,
                safety_two_se=0.01,
            )
            for size in plot.COHORTS
            for method in ("ppo", "pspo")
            for channel, step in [
                ("periodic", 0),
                (
                    "periodic",
                    180000 if size in (64, 128) and method == "pspo" else 200000,
                ),
                ("final", plot.STEPS[size, method]),
            ]
        ]
        self.rows = rows
        self.fig, self.axes = plot.build_figure(self.rows)

    def tearDown(self):
        plot.curves.plt.close(self.fig)

    def test_half_a4_dimensions_and_no_suptitle(self):
        self.assertEqual(self.axes.shape, (2, 4))
        self.assertLessEqual(self.fig.get_figwidth() * 25.4, 105)
        self.assertIsNone(self.fig._suptitle)
        self.assertEqual(self.fig.legends[0].get_title().get_text(), "")

    def test_exact_labels_and_titles(self):
        self.assertEqual(self.axes[0, 0].get_ylabel(), "Total reward")
        self.assertEqual(self.axes[1, 0].get_ylabel(), "Safety rate")
        self.assertEqual(self.fig._supxlabel.get_text(), "Training steps (thousands)")
        self.assertEqual(
            [t.get_text() for t in self.fig.legends[0].texts],
            ["PPO", "PSPO-VF", "Test-time evaluation"],
        )
        for column, size in enumerate(plot.COHORTS):
            self.assertEqual(
                self.axes[0, column].get_title().replace("\n", " "),
                f"FrozenLake {size}×{size}",
            )
            self.assertEqual(self.axes[1, column].get_title(), "")

    def test_black_ppo_green_pspo_and_correct_row_metrics(self):
        for row, expected in [(0, 0.2), (1, 0.9)]:
            for ax in self.axes[row]:
                lines = {
                    line.get_label(): line
                    for line in ax.lines
                    if line.get_label() in ("PPO", "PSPO-VF")
                }
                self.assertEqual(lines["PPO"].get_color(), "#000000")
                self.assertEqual(lines["PSPO-VF"].get_color(), "#009E73")
                self.assertEqual(
                    lines["PPO"].get_ydata().tolist(), [expected, expected]
                )

    def test_every_curve_uses_the_200k_budget(self):
        self.assertEqual(
            plot.COHORTS[128]["ppo"], "frozenlake128_shaping_ppo_t200000_20261008T210643Z"
        )
        self.assertEqual(set(plot.STEPS.values()), {204800, 200704})
        for ax in self.axes[:, 3]:
            self.assertEqual(len(ax.containers), 2)  # PPO and PSPO test-time points.
            for container in ax.containers:
                self.assertAlmostEqual(container.lines[0].get_xdata()[0], 200.704)

    def test_each_final_marker_is_connected_to_the_previous_recorded_point(self):
        for column, size in enumerate(plot.COHORTS):
            for row_number, metric in enumerate(("reward", "safety")):
                ax = self.axes[row_number, column]
                connectors = {
                    line.get_label(): line
                    for line in ax.lines
                    if line.get_label().startswith("_final_connector_")
                }
                for method in ("ppo", "pspo"):
                    finals = [
                        r
                        for r in self.rows
                        if r["size"] == size
                        and r["method"] == method
                        and r["channel"] == "final"
                    ]
                    label = f"_final_connector_{method}"
                    if not finals:
                        self.assertNotIn(label, connectors)
                        continue
                    last = max(
                        (
                            r
                            for r in self.rows
                            if r["size"] == size
                            and r["method"] == method
                            and r["channel"] == "periodic"
                        ),
                        key=lambda r: r["timestep"],
                    )
                    line = connectors[label]
                    self.assertEqual(
                        line.get_xdata().tolist(),
                        [last["timestep"] / 1000, finals[0]["timestep"] / 1000],
                    )
                    self.assertEqual(
                        line.get_ydata().tolist(),
                        [last[f"{metric}_mean"], finals[0][f"{metric}_mean"]],
                    )
                    self.assertEqual(line.get_color(), plot.COLORS[method])
                    self.assertEqual(
                        line.get_linestyle(), "--" if method == "ppo" else "-"
                    )


if __name__ == "__main__":
    unittest.main()
