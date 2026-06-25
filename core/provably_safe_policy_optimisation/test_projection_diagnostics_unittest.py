"""Tests for projection diagnostics (ProjectionResult, ProjectedAdam counters,
per-window SB3 logging)."""

from __future__ import annotations

import math
import types
import unittest

import gymnasium as gym
import torch as th
from torch import nn

from provably_safe_policy_optimisation.projected_optimizers import ProjectedAdam
from provably_safe_policy_optimisation.projected_dqn import ProjectedDQN
from provably_safe_policy_optimisation._projection_logging import record_projection_window
from provably_safe_policy_optimisation.projection import project_to_interval_union


class ProjectionResultTests(unittest.TestCase):
    def test_single_box_result_fields(self) -> None:
        x = nn.Parameter(th.tensor([5.0, -5.0]))
        res = project_to_interval_union(
            [x], [[th.tensor([-1.0, -1.0])]], [[th.tensor([1.0, 1.0])]]
        )
        self.assertEqual(res.n_projected, 2)          # both entries were outside
        self.assertEqual(res.n_boundary, 2)           # both now on a face
        self.assertEqual(res.selected_set_index, 0)
        self.assertAlmostEqual(res.displacement_l2, math.sqrt(32), places=5)
        self.assertAlmostEqual(res.displacement_linf, 4.0, places=5)
        self.assertTrue(th.allclose(x.detach(), th.tensor([1.0, -1.0])))

    def test_union_selects_nearest_box(self) -> None:
        x = nn.Parameter(th.tensor([10.0, 10.0]))
        res = project_to_interval_union(
            [x],
            [[th.tensor([0.0, 0.0])], [th.tensor([8.0, 8.0])]],
            [[th.tensor([1.0, 1.0])], [th.tensor([9.0, 9.0])]],
        )
        self.assertEqual(res.selected_set_index, 1)
        self.assertEqual(res.n_projected, 2)
        self.assertAlmostEqual(res.displacement_linf, 1.0, places=5)
        self.assertTrue(th.allclose(x.detach(), th.tensor([9.0, 9.0])))

    def test_no_clamp_when_inside(self) -> None:
        x = nn.Parameter(th.tensor([0.5]))
        res = project_to_interval_union([x], [[th.tensor([-1.0])]], [[th.tensor([1.0])]])
        self.assertEqual(res.n_projected, 0)
        self.assertEqual(res.displacement_l2, 0.0)


class ProjectedAdamDiagnosticsTests(unittest.TestCase):
    def test_active_fraction_and_means(self) -> None:
        x = nn.Parameter(th.tensor([5.0]))   # starts outside the box
        opt = ProjectedAdam([x], lr=1e-3)
        opt.set_bounds([th.tensor([-1.0])], [th.tensor([1.0])], project_on_set=False)

        x.grad = th.zeros_like(x)            # Adam no-op; isolate projection
        opt.step()                          # step 1: clamps 5 -> 1 (active)
        opt.step()                          # step 2: already in box (inactive)

        diag = opt.projection_diagnostics()
        self.assertEqual(diag["bounded_steps"], 2)
        self.assertEqual(diag["projection_active_steps"], 1)
        self.assertAlmostEqual(diag["projection_active_fraction"], 0.5)
        self.assertEqual(diag["projected_elements_total"], 1)
        self.assertAlmostEqual(diag["mean_projected_elements_per_step"], 0.5)
        self.assertAlmostEqual(diag["mean_projected_elements_per_active_step"], 1.0)
        self.assertEqual(diag["constrained_element_count"], 1)
        self.assertAlmostEqual(diag["mean_projected_fraction_per_step"], 0.5)
        self.assertGreater(diag["max_displacement_l2"], 0.0)
        self.assertEqual(diag["selected_box_counts"], {0: 2})

    def test_reset(self) -> None:
        x = nn.Parameter(th.tensor([5.0]))
        opt = ProjectedAdam([x], lr=1e-3)
        opt.set_bounds([th.tensor([-1.0])], [th.tensor([1.0])], project_on_set=False)
        x.grad = th.zeros_like(x)
        opt.step()
        opt.reset_projection_diagnostics()
        diag = opt.projection_diagnostics()
        self.assertEqual(diag["bounded_steps"], 0)
        self.assertEqual(diag["projected_elements_total"], 0)
        self.assertEqual(diag["selected_box_counts"], {})

    def test_no_bounds_has_empty_diagnostics(self) -> None:
        x = nn.Parameter(th.tensor([5.0]))
        opt = ProjectedAdam([x], lr=1e-3)
        x.grad = th.zeros_like(x)
        opt.step()  # no bounds -> not counted
        self.assertEqual(opt.projection_diagnostics()["bounded_steps"], 0)


class _FakeLogger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.values[key] = value


class ProjectionLoggingTests(unittest.TestCase):
    def test_record_projection_window_logs_deltas(self) -> None:
        x = nn.Parameter(th.tensor([5.0]))
        opt = ProjectedAdam([x], lr=1e-3)
        opt.set_bounds([th.tensor([-1.0])], [th.tensor([1.0])], project_on_set=False)
        logger = _FakeLogger()
        model = types.SimpleNamespace(
            policy=types.SimpleNamespace(optimizer=opt), logger=logger
        )

        x.grad = th.zeros_like(x)
        opt.step()
        record_projection_window(model)          # first call: seeds prev, no records
        self.assertEqual(logger.values, {})

        x.data = th.tensor([9.0])                 # force another clamp next step
        opt.step()
        record_projection_window(model)          # second call: logs the window
        self.assertIn("projection/active_step_fraction", logger.values)
        self.assertIn("projection/mean_displacement_l2", logger.values)
        self.assertEqual(logger.values["projection/active_step_fraction"], 1.0)


class ProjectedDQNDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_passthrough_after_learn(self) -> None:
        env = gym.make("CartPole-v1")
        probe = ProjectedDQN(
            "MlpPolicy", env, seed=0, device="cpu", verbose=0,
            learning_starts=50, buffer_size=1_000, batch_size=16,
            train_freq=1, target_update_interval=10, policy_kwargs={"net_arch": [16]},
        )
        params = list(probe.policy.q_net.parameters())
        lower = [p.detach() - 0.01 for p in params]
        upper = [p.detach() + 0.01 for p in params]
        model = ProjectedDQN(
            "MlpPolicy", gym.make("CartPole-v1"), seed=0, device="cpu", verbose=0,
            learning_starts=50, buffer_size=1_000, batch_size=16, learning_rate=1e-2,
            train_freq=1, target_update_interval=10, policy_kwargs={"net_arch": [16]},
            param_bounds_l=lower, param_bounds_u=upper,
        )
        self.addCleanup(model.get_env().close)
        self.addCleanup(probe.get_env().close)
        model.learn(total_timesteps=400)

        diag = model.projection_diagnostics()
        self.assertGreater(diag["bounded_steps"], 0)
        self.assertGreaterEqual(diag["projection_active_fraction"], 0.0)
        self.assertLessEqual(diag["projection_active_fraction"], 1.0)
        self.assertGreater(diag["projected_elements_total"], 0)


if __name__ == "__main__":
    unittest.main()
