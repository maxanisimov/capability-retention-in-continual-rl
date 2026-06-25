"""Unit and integration tests for projected-gradient-descent DQN.

Covers :class:`ProjectedAdam` (the projection optimizer) and
:class:`ProjectedDQN` (the SB3-integrated training class).
"""

from __future__ import annotations

import unittest

import gymnasium as gym
import torch as th
from torch import nn

from experiments.utils.projected_optimizers import ProjectedAdam
from experiments.utils.sb3_projected_dqn import ProjectedDQN


def _tiny_module(*shapes: tuple[int, ...]) -> nn.ParameterList:
    """A module whose parameters have the requested shapes (values=ones)."""
    return nn.ParameterList([nn.Parameter(th.ones(*s)) for s in shapes])


class ProjectedAdamTests(unittest.TestCase):
    def test_no_bounds_matches_plain_adam(self) -> None:
        """Without bounds, ProjectedAdam must be identical to torch.optim.Adam."""
        th.manual_seed(0)
        ref = nn.Linear(4, 3)
        proj = nn.Linear(4, 3)
        proj.load_state_dict(ref.state_dict())

        opt_ref = th.optim.Adam(ref.parameters(), lr=1e-2)
        opt_proj = ProjectedAdam(proj.parameters(), lr=1e-2)
        self.assertFalse(opt_proj.has_bounds)

        x = th.randn(8, 4)
        for _ in range(5):
            for opt, net in ((opt_ref, ref), (opt_proj, proj)):
                opt.zero_grad()
                net(x).pow(2).mean().backward()
                opt.step()

        for p_ref, p_proj in zip(ref.parameters(), proj.parameters()):
            self.assertTrue(th.allclose(p_ref, p_proj, atol=1e-7))

    def test_single_box_keeps_params_inside(self) -> None:
        net = nn.Linear(4, 3)
        opt = ProjectedAdam(net.parameters(), lr=0.5)
        params = list(net.parameters())
        lower = [th.full_like(p, -0.1) for p in params]
        upper = [th.full_like(p, 0.1) for p in params]
        opt.set_bounds(lower, upper)
        self.assertTrue(opt.has_bounds)

        x = th.randn(16, 4)
        for _ in range(10):
            opt.zero_grad()
            net(x).pow(2).mean().backward()
            opt.step()

        for p, lo, hi in zip(net.parameters(), lower, upper):
            self.assertTrue(th.all(p >= lo - 1e-6))
            self.assertTrue(th.all(p <= hi + 1e-6))

    def test_union_projects_to_nearest_box_l2(self) -> None:
        module = _tiny_module((2,))
        with th.no_grad():
            module[0].copy_(th.tensor([10.0, 10.0]))
        opt = ProjectedAdam(module.parameters(), lr=1e-3)
        # Box A is far ([0,1]), Box B is closer ([5,6]); interval-major bounds.
        bounds_l = [[th.tensor([0.0, 0.0])], [th.tensor([5.0, 5.0])]]
        bounds_u = [[th.tensor([1.0, 1.0])], [th.tensor([6.0, 6.0])]]
        opt.set_bounds(bounds_l, bounds_u)

        # Zero gradient => Adam leaves the param untouched; only projection acts.
        module[0].grad = th.zeros_like(module[0])
        opt.step()

        self.assertTrue(th.allclose(module[0].data, th.tensor([6.0, 6.0])))

    def test_all_norms_pick_per_coordinate_dominating_box(self) -> None:
        bounds_l = [[th.tensor([0.0, 0.0])], [th.tensor([8.0, 8.0])]]
        bounds_u = [[th.tensor([1.0, 1.0])], [th.tensor([9.0, 9.0])]]
        for norm in ("l2", "l1", "linf"):
            module = _tiny_module((2,))
            with th.no_grad():
                module[0].copy_(th.tensor([10.0, 10.0]))
            opt = ProjectedAdam(module.parameters(), lr=1e-3, distance_norm=norm)
            opt.set_bounds(bounds_l, bounds_u)
            module[0].grad = th.zeros_like(module[0])
            opt.step()
            self.assertTrue(
                th.allclose(module[0].data, th.tensor([9.0, 9.0])),
                msg=f"norm={norm} did not project to the dominating box",
            )

    def test_projection_counters_increment(self) -> None:
        net = nn.Linear(2, 2)
        opt = ProjectedAdam(net.parameters(), lr=0.5)
        opt.set_bounds(
            [th.full_like(p, -0.01) for p in net.parameters()],
            [th.full_like(p, 0.01) for p in net.parameters()],
        )
        x = th.randn(4, 2)
        for _ in range(3):
            opt.zero_grad()
            net(x).pow(2).mean().backward()
            opt.step()
        self.assertEqual(opt._step_calls, 3)
        self.assertGreater(opt._projected_elements, 0)

    def test_set_bounds_shape_mismatch_raises(self) -> None:
        net = nn.Linear(4, 3)
        opt = ProjectedAdam(net.parameters(), lr=1e-2)
        params = list(net.parameters())
        bad_lower = [th.zeros(99) for _ in params]  # wrong shapes
        bad_upper = [th.ones(99) for _ in params]
        with self.assertRaises(ValueError):
            opt.set_bounds(bad_lower, bad_upper)


class ProjectedDQNTests(unittest.TestCase):
    def _make_model(self, **bounds_kwargs) -> ProjectedDQN:
        env = gym.make("CartPole-v1")
        model = ProjectedDQN(
            "MlpPolicy",
            env,
            learning_starts=50,
            buffer_size=1_000,
            batch_size=16,
            train_freq=1,
            target_update_interval=10,
            seed=0,
            device="cpu",
            policy_kwargs={"net_arch": [16]},
            **bounds_kwargs,
        )
        self.addCleanup(model.get_env().close)
        return model

    def test_injects_projected_optimizer(self) -> None:
        model = self._make_model()
        self.assertIsInstance(model.policy.optimizer, ProjectedAdam)
        self.assertFalse(model.policy.optimizer.has_bounds)

    def test_presupplied_bounds_attach_and_train(self) -> None:
        # Build once to discover q_net parameter shapes/order.
        probe = self._make_model()
        wide_l = [th.full_like(p, -1e9) for p in probe.policy.q_net.parameters()]
        wide_u = [th.full_like(p, 1e9) for p in probe.policy.q_net.parameters()]

        model = self._make_model(param_bounds_l=wide_l, param_bounds_u=wide_u)
        self.assertTrue(model.policy.optimizer.has_bounds)
        model.learn(total_timesteps=300)
        self.assertGreater(model.policy.optimizer._step_calls, 0)

    def test_misaligned_bounds_raise_at_construction(self) -> None:
        probe = self._make_model()
        n_params = len(list(probe.policy.q_net.parameters()))
        bad_l = [th.zeros(7) for _ in range(n_params)]  # wrong shapes
        bad_u = [th.ones(7) for _ in range(n_params)]
        with self.assertRaises(ValueError):
            self._make_model(param_bounds_l=bad_l, param_bounds_u=bad_u)

    def test_params_stay_in_tight_box_during_training(self) -> None:
        model = self._make_model()
        params = list(model.policy.q_net.parameters())
        # Tight box around the initial weights.
        lower = [p.detach().clone() - 0.02 for p in params]
        upper = [p.detach().clone() + 0.02 for p in params]
        model.policy.optimizer.set_bounds(lower, upper)

        model.learn(total_timesteps=400)

        for p, lo, hi in zip(model.policy.q_net.parameters(), lower, upper):
            self.assertTrue(th.all(p >= lo - 1e-6), msg="param fell below lower bound")
            self.assertTrue(th.all(p <= hi + 1e-6), msg="param exceeded upper bound")
        self.assertGreater(model.policy.optimizer._projected_elements, 0)


if __name__ == "__main__":
    unittest.main()
