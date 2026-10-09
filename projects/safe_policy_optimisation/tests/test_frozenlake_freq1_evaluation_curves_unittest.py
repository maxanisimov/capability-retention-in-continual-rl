"""Seed counts, unmodified returns and endpoint semantics for the new plots."""

import unittest

from projects.safe_policy_optimisation.scripts import (
    plot_frozenlake_freq1_evaluation_curves as plot,
)


class FrequencyOneCurveTests(unittest.TestCase):
    def test_ppo128_trim_keeps_initialisation_and_does_not_move_final(self):
        points = [
            dict(size=size, method=method, channel=channel, timestep=step)
            for size, method, channel, step in [
                (128, "ppo", "periodic", 0),
                (128, "ppo", "periodic", 200000),
                (128, "ppo", "periodic", 220000),
                (128, "ppo", "final", 401408),
                (128, "pspo", "final", 200704),
                (16, "ppo", "final", 204800),
            ]
        ]
        self.assertEqual(plot.trim_ppo128(points), [points[i] for i in (0, 1, 4, 5)])
        self.assertEqual(points[3]["timestep"], 401408)
        with self.assertRaises(ValueError):
            plot.trim_ppo128(points, 0)

    def test_pspo_periodic_stop_uses_requested_not_rounded_budget(self):
        periodic = [
            dict(
                timestep=step,
                episodes=10,
                mean_total_reward=0.9,
                success_rate=1,
                safety_rate=1,
            )
            for step in [*range(20000, 200000, 20000), 200704]
        ]
        initial = dict(episodes=10, reward=-2.5, goal=0, safety=1)
        final = dict(episodes=100, reward=0.8, goal=1, safety=1)
        points = plot.evaluation_points(
            periodic, initial, final, 200704, periodic_stop=200000
        )
        self.assertNotIn(200000, [p["timestep"] for p in points])
        self.assertEqual(points[-1]["timestep"], 200704)
        with self.assertRaises(ValueError):
            plot.evaluation_points(periodic, initial, final, 200704)
        with self.assertRaises(ValueError):
            plot.evaluation_points(
                periodic[:-2] + periodic[-1:],
                initial,
                final,
                200704,
                periodic_stop=200000,
            )

    def test_four_layout_cohorts_use_matched_short64_and_frequency1_128(self):
        self.assertEqual(list(plot.FOUR_LAYOUT_COHORTS), [16, 32, 64, 128])
        for size in plot.FOUR_LAYOUT_COHORTS:
            self.assertIn("freq1", plot.FOUR_LAYOUT_COHORTS[size]["pspo"])
            for method in ("ppo", "pspo"):
                self.assertEqual(plot.FOUR_LAYOUT_SEEDS[size, method], list(range(10)))
        self.assertEqual(plot.FOUR_LAYOUT_STEPS[64, "ppo"], 200704)
        self.assertEqual(plot.FOUR_LAYOUT_STEPS[64, "pspo"], 200704)
        self.assertEqual(plot.FOUR_LAYOUT_STEPS[128, "ppo"], 401408)
        self.assertEqual(plot.FOUR_LAYOUT_STEPS[128, "pspo"], 200704)

    def test_four_layout_figure_keeps_trimmed_prefix_and_real_pspo_endpoint(self):
        rows = [
            dict(
                size=size,
                method=method,
                channel=channel,
                timestep=step,
                seed_count=10,
                reward_mean=0.9,
                reward_two_se=0.03,
            )
            for size in plot.FOUR_LAYOUT_COHORTS
            for method in ("ppo", "pspo")
            for channel, step in [
                ("periodic", 0),
                ("periodic", 200000),
                ("final", plot.FOUR_LAYOUT_STEPS[size, method]),
            ]
        ]
        rows = plot.trim_ppo128(rows)
        fig, axes = plot.build_reward_figure(rows)
        try:
            self.assertEqual(len(axes), 4)
            self.assertAlmostEqual(fig.get_figwidth(), 7.1)
            self.assertIn("PPO prefix ≤200k", axes[3].get_title())
            self.assertEqual(len(axes[3].containers), 1)
            self.assertAlmostEqual(
                axes[3].containers[0].lines[0].get_xdata()[0], 200.704
            )
            self.assertLess(axes[3].get_xlim()[1], 220)
        finally:
            plot.plt.close(fig)

    def test_four_layout_aggregate_rejects_missing_seed128(self):
        points = [
            dict(
                size=128,
                method="pspo",
                seed=seed,
                channel="periodic",
                timestep=20000,
                reward=1,
                goal=1,
                safety=1,
            )
            for seed in range(10)
        ]
        self.assertEqual(
            plot.aggregate(points, expected_seeds=plot.FOUR_LAYOUT_SEEDS)[0][
                "seed_count"
            ],
            10,
        )
        with self.assertRaises(ValueError):
            plot.aggregate(points[:-1], expected_seeds=plot.FOUR_LAYOUT_SEEDS)

    def test_crop_removes_long_run_final_instead_of_relabelling_it(self):
        from projects.safe_policy_optimisation.scripts import (
            plot_frozenlake_freq1_reward_safety_curves as joint,
        )

        points = [
            dict(size=size, timestep=step, channel=channel)
            for size, step, channel in [
                (32, 204800, "final"),
                (64, 200000, "periodic"),
                (64, 409600, "final"),
            ]
        ]
        cropped = joint.trim_layout64(points, 200000)
        self.assertEqual(cropped, points[:2])
        self.assertEqual(points[-1]["timestep"], 409600)
        with self.assertRaises(ValueError):
            joint.trim_layout64(points, 0)

    def test_matched_ppo64_requires_all_ten_seeds(self):
        points = [
            dict(
                size=64,
                method="ppo",
                seed=seed,
                channel="final",
                timestep=200704,
                reward=1,
                goal=1,
                safety=1,
            )
            for seed in range(10)
        ]
        self.assertEqual(plot.aggregate(points, matched64=True)[0]["seed_count"], 10)
        with self.assertRaises(ValueError):
            plot.aggregate(points[:1], matched64=True)

    def test_cropped_panel_has_no_fabricated_final_evaluation(self):
        rows = [
            dict(
                size=64,
                method=method,
                channel="periodic",
                timestep=step,
                seed_count=10,
                reward_mean=1,
                reward_two_se=0,
                safety_mean=1,
                safety_two_se=0,
            )
            for method in ("ppo", "pspo")
            for step in (0, 200000)
        ]
        fig, ax = plot.plt.subplots()
        try:
            plot.draw_panel(ax, rows, 64)
            self.assertLess(ax.get_xlim()[1], 220)
            self.assertEqual(len(ax.containers), 0)
            self.assertIn("Ten seeds", ax.get_title())
        finally:
            plot.plt.close(fig)

    def test_standard_return_uses_goal_not_penalised_reward(self):
        points = [
            dict(reward=-2.5, goal=0),
            dict(reward=0.447257, goal=0.971),
            dict(reward=-0.5, goal=1),
        ]
        corrected = plot.with_reward_definition(points)
        self.assertEqual([p["reward"] for p in corrected], [0, 0.971, 1])
        self.assertEqual(
            [p["step_penalised_reward"] for p in corrected], [-2.5, 0.447257, -0.5]
        )
        self.assertEqual(points[1]["reward"], 0.447257)
        restored = plot.with_reward_definition(corrected, "step-penalised")
        self.assertEqual([p["reward"] for p in restored], [-2.5, 0.447257, -0.5])

    def test_standard_return_se_recomputed_from_seed_success(self):
        points = [
            dict(
                size=16,
                method="pspo",
                channel="final",
                timestep=204800,
                seed=seed,
                reward=0.9 - seed * 0.01,
                goal=1,
                safety=1,
            )
            for seed in range(10)
        ]
        row = plot.aggregate(plot.with_reward_definition(points))[0]
        self.assertEqual(row["reward_mean"], 1)
        self.assertEqual(row["reward_two_se"], 0)
        self.assertGreater(row["step_penalised_reward_two_se"], 0)

    def test_invalid_goal_values_are_rejected_not_clipped(self):
        for value in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                plot.with_reward_definition([dict(reward=0, goal=value)])

    def test_single_seed_has_no_standard_error(self):
        self.assertEqual(
            plot.seed_statistics([0.8]), {"mean": 0.8, "two_se": None, "seed_count": 1}
        )

    def test_two_sample_standard_errors(self):
        self.assertAlmostEqual(plot.seed_statistics([1, 3])["two_se"], 2)

    def test_final_100_episode_score_separate_and_no_extrapolation(self):
        periodic = [
            dict(
                timestep=20000,
                episodes=10,
                mean_total_reward=0.6,
                success_rate=1,
                safety_rate=1,
            ),
            dict(
                timestep=20480,
                episodes=10,
                mean_total_reward=0.7,
                success_rate=1,
                safety_rate=1,
            ),
        ]
        initial = dict(episodes=10, reward=-0.625, goal=0, safety=1)
        final = dict(episodes=100, reward=0.9, goal=1, safety=1)
        points = plot.evaluation_points(periodic, initial, final, 20480)
        self.assertEqual([p["timestep"] for p in points], [0, 20000, 20480])
        self.assertEqual([p["reward"] for p in points], [-0.625, 0.6, 0.9])
        self.assertEqual(points[-1]["channel"], "final")
        with self.assertRaises(ValueError):
            plot.evaluation_points(periodic, initial, final, 40960)

    def test_matched_ten_seeds_required(self):
        points = [
            dict(
                size=16,
                method="pspo",
                channel="periodic",
                timestep=20000,
                seed=seed,
                reward=0.8,
                goal=1,
                safety=1,
            )
            for seed in range(10)
        ]
        self.assertEqual(plot.aggregate(points)[0]["seed_count"], 10)
        with self.assertRaises(ValueError):
            plot.aggregate(points[:-1])
        points[-1]["seed"] = 0
        with self.assertRaises(ValueError):
            plot.aggregate(points)

    def test_ppo64_pilot_kept_as_one_seed(self):
        point = dict(
            size=64,
            method="ppo",
            channel="final",
            timestep=200704,
            seed=0,
            reward=0.78751,
            goal=0.99,
            safety=0.99,
        )
        row = plot.aggregate([point])[0]
        self.assertEqual(row["seed_count"], 1)
        self.assertIsNone(row["reward_two_se"])
        self.assertEqual(row["timestep"], 200704)

    def test_safety_panel_uses_safety_not_reward(self):
        rows = [
            dict(
                size=16,
                method=method,
                channel=channel,
                timestep=step,
                seed_count=10,
                reward_mean=0.2,
                reward_two_se=0.02,
                safety_mean=0.9,
                safety_two_se=0.01,
            )
            for method in ("ppo", "pspo")
            for channel, step in [
                ("periodic", 0),
                ("periodic", 20000),
                ("final", 204800),
            ]
        ]
        for metric, expected in [("reward", 0.2), ("safety", 0.9)]:
            fig, ax = plot.plt.subplots()
            try:
                plot.draw_panel(ax, rows, 16, metric=metric)
                self.assertEqual(
                    ax.get_lines()[0].get_ydata().tolist(), [expected, expected]
                )
            finally:
                plot.plt.close(fig)

    def test_joint_figure_contains_both_metrics_for_all_three_layouts(self):
        from projects.safe_policy_optimisation.scripts import (
            plot_frozenlake_freq1_reward_safety_curves as joint,
        )

        rows = [
            dict(
                size=size,
                method=method,
                channel=channel,
                timestep=step,
                seed_count=1 if size == 64 and method == "ppo" else 10,
                reward_mean=0.2,
                reward_two_se=None if size == 64 and method == "ppo" else 0.02,
                safety_mean=0.9,
                safety_two_se=None if size == 64 and method == "ppo" else 0.01,
            )
            for size in (16, 32, 64)
            for method in ("ppo", "pspo")
            for channel, step in [
                ("periodic", 0),
                ("periodic", 20000),
                ("final", plot.EXPECTED_STEPS[size, method]),
            ]
        ]
        fig, axes = joint.build_figure(rows)
        try:
            self.assertEqual(axes.shape, (2, 3))
            self.assertEqual(len(fig.axes), 6)
            self.assertEqual(axes[0, 0].get_ylabel(), "Standard total reward")
            self.assertEqual(axes[1, 0].get_ylabel(), "Safety rate")
            for column in range(3):
                self.assertEqual(
                    axes[0, column].get_lines()[0].get_ydata().tolist(), [0.2, 0.2]
                )
                self.assertEqual(
                    axes[1, column].get_lines()[0].get_ydata().tolist(), [0.9, 0.9]
                )
        finally:
            plot.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
