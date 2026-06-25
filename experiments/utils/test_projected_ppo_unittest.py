"""Unit and integration tests for projected-gradient-descent PPO.

Covers the param-subset projection added to :class:`ProjectedAdam` and the
SB3-integrated :class:`ProjectedPPO` training class.
"""

from __future__ import annotations

import unittest

import gymnasium as gym
import torch as th
from torch import nn

from experiments.utils.projected_optimizers import ProjectedAdam
from experiments.utils.sb3_projected_ppo import (
    ProjectedPPO,
    projection_target_parameter_names,
)


class ProjectedAdamSubsetTests(unittest.TestCase):
    def test_projects_only_the_requested_subset(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        first, second = net[0], net[1]
        opt = ProjectedAdam(net.parameters(), lr=0.5)

        # Constrain only the first layer's parameters to a tight box.
        target = list(first.parameters())
        lower = [th.full_like(p, -0.01) for p in target]
        upper = [th.full_like(p, 0.01) for p in target]
        opt.set_bounds(lower, upper, params=target)

        second_before = [p.detach().clone() for p in second.parameters()]
        x = th.randn(16, 4)
        for _ in range(8):
            opt.zero_grad()
            net(x).pow(2).mean().backward()
            opt.step()

        # First layer stays inside the box ...
        for p in first.parameters():
            self.assertTrue(th.all(p >= -0.01 - 1e-6))
            self.assertTrue(th.all(p <= 0.01 + 1e-6))
        # ... while the unconstrained second layer is free to move.
        moved = any(
            not th.allclose(b, a, atol=1e-6)
            for b, a in zip(second_before, second.parameters())
        )
        self.assertTrue(moved, "unconstrained params should still update")

    def test_subset_must_belong_to_optimizer(self) -> None:
        net = nn.Linear(4, 2)
        stranger = nn.Linear(4, 2)  # not owned by the optimizer
        opt = ProjectedAdam(net.parameters(), lr=0.1)
        foreign = list(stranger.parameters())
        with self.assertRaises(ValueError):
            opt.set_bounds(
                [th.zeros_like(p) for p in foreign],
                [th.ones_like(p) for p in foreign],
                params=foreign,
            )


class ProjectedPPOTests(unittest.TestCase):
    def _make_model(self, **kwargs) -> ProjectedPPO:
        env = gym.make("CartPole-v1")
        model = ProjectedPPO(
            "MlpPolicy",
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            seed=0,
            device="cpu",
            policy_kwargs={"net_arch": [16]},
            **kwargs,
        )
        self.addCleanup(model.get_env().close)
        return model

    def test_injects_projected_optimizer(self) -> None:
        model = self._make_model()
        self.assertIsInstance(model.policy.optimizer, ProjectedAdam)
        self.assertFalse(model.policy.optimizer.has_bounds)

    def test_feature_actor_names_exclude_value_net(self) -> None:
        model = self._make_model()
        names = projection_target_parameter_names(model, projection_target="feature_actor")
        self.assertTrue(names)
        self.assertFalse(any(".value_net." in n or n.startswith("value_net.") for n in names))

    def test_misaligned_bounds_raise_at_construction(self) -> None:
        probe = self._make_model()
        names = projection_target_parameter_names(probe)
        bad_l = [th.zeros(7) for _ in names]  # wrong shapes
        bad_u = [th.ones(7) for _ in names]
        with self.assertRaises(ValueError):
            self._make_model(param_bounds_l=bad_l, param_bounds_u=bad_u)

    def test_actor_stays_in_box_critic_free(self) -> None:
        probe = self._make_model()
        names = projection_target_parameter_names(probe)
        name_to_param = dict(probe.policy.named_parameters())
        # Tight box around the actor's initial weights; a high LR ensures the
        # actor tries to move past the box so the projection actually fires.
        lower = [name_to_param[n].detach().clone() - 0.005 for n in names]
        upper = [name_to_param[n].detach().clone() + 0.005 for n in names]

        # Build a fresh model with the same seed so init weights match `names`.
        model = self._make_model(
            param_bounds_l=lower, param_bounds_u=upper, learning_rate=0.05
        )
        self.assertTrue(model.policy.optimizer.has_bounds)

        value_names = [n for n, _ in model.policy.named_parameters() if "value_net" in n]
        value_before = {n: dict(model.policy.named_parameters())[n].detach().clone() for n in value_names}

        model.learn(total_timesteps=256)

        final = dict(model.policy.named_parameters())
        for n, lo, hi in zip(names, lower, upper):
            self.assertTrue(th.all(final[n] >= lo - 1e-6), msg=f"{n} fell below lower bound")
            self.assertTrue(th.all(final[n] <= hi + 1e-6), msg=f"{n} exceeded upper bound")
        self.assertGreater(model.policy.optimizer._projected_elements, 0)
        # Critic (value_net) is unconstrained and should have moved.
        critic_moved = any(
            not th.allclose(final[n], value_before[n], atol=1e-6) for n in value_names
        )
        self.assertTrue(critic_moved, "value network should train unconstrained")


if __name__ == "__main__":
    unittest.main()
