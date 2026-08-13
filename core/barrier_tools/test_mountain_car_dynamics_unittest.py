"""Tests for exact MountainCarContinuous dynamics."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch

from barrier_tools.dynamics.mountain_car import MountainCarContinuousDynamics


class MountainCarContinuousDynamicsTests(unittest.TestCase):
    def test_step_matches_unwrapped_gymnasium_for_random_pairs(self) -> None:
        env = gym.make("MountainCarContinuous-v0").unwrapped
        dynamics = MountainCarContinuousDynamics()
        rng = np.random.default_rng(0)

        states = rng.uniform(
            low=np.array([-1.2, -0.07], dtype=np.float32),
            high=np.array([0.6, 0.07], dtype=np.float32),
            size=(10_000, 2),
        ).astype(np.float32)
        actions = rng.uniform(-2.0, 2.0, size=(10_000, 1)).astype(np.float32)

        special_states = np.array(
            [
                [-0.5, 0.0],
                [-0.5, 0.0699],
                [-0.5, -0.0699],
                [-1.2, -0.07],
                [0.6, 0.07],
            ],
            dtype=np.float32,
        )
        special_actions = np.array([[2.0], [1.0], [-1.0], [-2.0], [2.0]], dtype=np.float32)
        states[: len(special_states)] = special_states
        actions[: len(special_actions)] = special_actions

        custom = dynamics.step(torch.from_numpy(states), torch.from_numpy(actions)).numpy()
        expected = np.empty_like(custom)
        for idx, (state, action) in enumerate(zip(states, actions)):
            env.state = state.copy()
            expected[idx], *_ = env.step(action)

        np.testing.assert_allclose(custom, expected, rtol=0.0, atol=2e-7)

    def test_interval_step_contains_sampled_points(self) -> None:
        dynamics = MountainCarContinuousDynamics()
        lower = torch.tensor([[-1.1, -0.05]])
        upper = torch.tensor([[-0.9, 0.03]])
        action_l = torch.tensor([[-1.5]])
        action_u = torch.tensor([[0.7]])
        next_l, next_u = dynamics.interval_step(lower, upper, action_l, action_u)

        generator = torch.Generator().manual_seed(1)
        states = lower + torch.rand((512, 2), generator=generator) * (upper - lower)
        actions = action_l + torch.rand((512, 1), generator=generator) * (action_u - action_l)
        next_states = dynamics.step(states, actions)

        self.assertTrue(bool((next_states >= next_l - 1e-7).all().item()))
        self.assertTrue(bool((next_states <= next_u + 1e-7).all().item()))


if __name__ == "__main__":
    unittest.main()
