"""Tests for the feasibility guarantees: non-empty-box validation, initial
projection, on-demand projection, the within-bounds verification utility, and
the reload warning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import torch as th
from torch import nn

from provably_safe_policy_optimisation.projected_optimizers import ProjectedAdam
from provably_safe_policy_optimisation.projected_dqn import ProjectedDQN
from provably_safe_policy_optimisation.projected_ppo import ProjectedPPO
from provably_safe_policy_optimisation.projection import (
    validate_and_prepare_param_interval_bounds,
)


def _make_dqn(**extra) -> ProjectedDQN:
    return ProjectedDQN(
        "MlpPolicy", gym.make("CartPole-v1"), seed=0, device="cpu", verbose=0,
        learning_starts=50, buffer_size=1_000, batch_size=16,
        train_freq=1, target_update_interval=10, policy_kwargs={"net_arch": [16]},
        **extra,
    )


class EmptyBoxValidationTests(unittest.TestCase):
    def test_single_box_lb_gt_ub_raises(self) -> None:
        p = nn.Parameter(th.zeros(3))
        with self.assertRaises(ValueError):
            validate_and_prepare_param_interval_bounds(
                actor_params=[p],
                actor_param_bounds_l=[th.ones(3)],       # lower > upper
                actor_param_bounds_u=[-th.ones(3)],
                device=th.device("cpu"),
            )

    def test_union_box_lb_gt_ub_raises(self) -> None:
        p = nn.Parameter(th.zeros(2))
        with self.assertRaises(ValueError):
            validate_and_prepare_param_interval_bounds(
                actor_params=[p],
                actor_param_bounds_l=[[th.tensor([-1.0, -1.0])], [th.tensor([1.0, 1.0])]],
                actor_param_bounds_u=[[th.tensor([1.0, 1.0])], [th.tensor([0.0, 0.0])]],  # 2nd box bad
                device=th.device("cpu"),
            )


class InitProjectionAndVerificationTests(unittest.TestCase):
    def test_set_bounds_projects_initial_params(self) -> None:
        x = nn.Parameter(th.tensor([5.0, -5.0]))   # outside the box
        opt = ProjectedAdam([x], lr=1e-3)
        opt.set_bounds([th.tensor([-1.0, -1.0])], [th.tensor([1.0, 1.0])])  # project_on_set default
        self.assertTrue(opt.is_within_bounds())
        self.assertEqual(opt.max_violation(), 0.0)
        self.assertIsNotNone(opt._init_projection)
        self.assertEqual(opt._init_projection.n_projected, 2)
        self.assertTrue(th.allclose(x.detach(), th.tensor([1.0, -1.0])))

    def test_max_violation_value(self) -> None:
        x = nn.Parameter(th.tensor([0.0, 3.0]))
        opt = ProjectedAdam([x], lr=1e-3)
        opt.set_bounds([th.tensor([-1.0, -1.0])], [th.tensor([1.0, 1.0])], project_on_set=False)
        self.assertAlmostEqual(opt.max_violation(), 2.0)   # 3.0 exceeds ub=1.0 by 2.0
        self.assertFalse(opt.is_within_bounds())

    def test_project_now_restores_feasibility(self) -> None:
        x = nn.Parameter(th.tensor([0.5]))
        opt = ProjectedAdam([x], lr=1e-3)
        opt.set_bounds([th.tensor([-1.0])], [th.tensor([1.0])])
        x.data = th.tensor([9.0])               # manual out-of-bounds edit
        self.assertFalse(opt.is_within_bounds())
        opt.project_now()
        self.assertTrue(opt.is_within_bounds())

    def test_no_bounds_is_trivially_within(self) -> None:
        x = nn.Parameter(th.tensor([100.0]))
        opt = ProjectedAdam([x], lr=1e-3)
        self.assertTrue(opt.is_within_bounds())
        self.assertEqual(opt.max_violation(), 0.0)


class ModelFeasibilityTests(unittest.TestCase):
    def test_dqn_feasible_immediately_and_after_training(self) -> None:
        probe = _make_dqn()
        params = list(probe.policy.q_net.parameters())
        # A box that does NOT contain the initial weights (centred away from them).
        lower = [p.detach() + 1.0 for p in params]
        upper = [p.detach() + 2.0 for p in params]
        self.addCleanup(probe.get_env().close)

        model = _make_dqn(learning_rate=1e-2, param_bounds_l=lower, param_bounds_u=upper)
        self.addCleanup(model.get_env().close)
        self.assertTrue(model.is_within_bounds())          # feasible before any training
        model.learn(total_timesteps=400)
        self.assertTrue(model.is_within_bounds(atol=1e-6)) # and after

    def test_ppo_feasible_immediately(self) -> None:
        probe = ProjectedPPO(
            "MlpPolicy", gym.make("CartPole-v1"), seed=0, device="cpu", verbose=0,
            n_steps=64, batch_size=32, n_epochs=1, policy_kwargs={"net_arch": [16]},
        )
        self.addCleanup(probe.get_env().close)
        from provably_safe_policy_optimisation import projection_target_parameter_names
        names = projection_target_parameter_names(probe)
        p0 = dict(probe.policy.named_parameters())
        lower = [p0[n].detach() + 1.0 for n in names]      # box excludes init
        upper = [p0[n].detach() + 2.0 for n in names]
        model = ProjectedPPO(
            "MlpPolicy", gym.make("CartPole-v1"), seed=0, device="cpu", verbose=0,
            n_steps=64, batch_size=32, n_epochs=1, policy_kwargs={"net_arch": [16]},
            param_bounds_l=lower, param_bounds_u=upper,
        )
        self.addCleanup(model.get_env().close)
        self.assertTrue(model.is_within_bounds())


class ReloadTests(unittest.TestCase):
    def test_constructor_bounds_survive_reload_without_warning(self) -> None:
        """Bounds supplied via the constructor are pickled and re-applied on load."""
        import warnings

        probe = _make_dqn()
        self.addCleanup(probe.get_env().close)
        params = list(probe.policy.q_net.parameters())
        lower = [p.detach() - 0.05 for p in params]
        upper = [p.detach() + 0.05 for p in params]
        model = _make_dqn(param_bounds_l=lower, param_bounds_u=upper)
        self.addCleanup(model.get_env().close)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.zip"
            model.save(path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reloaded = ProjectedDQN.load(path, env=gym.make("CartPole-v1"))
            self.addCleanup(reloaded.get_env().close)

        projection_warnings = [w for w in caught if "projection" in str(w.message).lower()]
        self.assertEqual(projection_warnings, [])               # no warning
        self.assertTrue(reloaded.policy.optimizer.has_bounds)   # bounds restored
        self.assertTrue(reloaded.is_within_bounds())

    def test_escape_hatch_bounds_warn_on_reload_and_reattach(self) -> None:
        """Bounds attached directly on the optimizer are not persisted -> warn."""
        model = _make_dqn()   # no constructor bounds
        self.addCleanup(model.get_env().close)
        params = list(model.policy.q_net.parameters())
        model.policy.optimizer.set_bounds(
            [p.detach() - 0.05 for p in params], [p.detach() + 0.05 for p in params]
        )

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.zip"
            model.save(path)
            with self.assertWarns(UserWarning):
                reloaded = ProjectedDQN.load(path, env=gym.make("CartPole-v1"))
            self.addCleanup(reloaded.get_env().close)

            self.assertFalse(reloaded.policy.optimizer.has_bounds)
            r_params = list(reloaded.policy.q_net.parameters())
            reloaded.set_projection_bounds(
                [p.detach() - 0.05 for p in r_params],
                [p.detach() + 0.05 for p in r_params],
            )
            self.assertTrue(reloaded.policy.optimizer.has_bounds)
            self.assertTrue(reloaded.is_within_bounds())


if __name__ == "__main__":
    unittest.main()
