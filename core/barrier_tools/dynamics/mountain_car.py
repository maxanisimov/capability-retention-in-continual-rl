"""Exact dynamics for Gymnasium's ``MountainCarContinuous-v0``."""

from __future__ import annotations

import torch

from barrier_tools.verification.interval import clip_interval, cos_interval


class MountainCarContinuousDynamics:
    """One-step model matching Gymnasium's continuous MountainCar implementation."""

    state_dim = 2
    action_dim = 1

    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    min_action = -1.0
    max_action = 1.0
    power = 0.0015
    gravity = 0.0025

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the next state for a point state/action batch."""

        if state.shape[-1] != 2:
            raise ValueError("MountainCar state must have final dimension 2.")
        if action.shape[-1] != 1:
            action = action.reshape(*state.shape[:-1], 1)

        position = state[..., 0]
        velocity = state[..., 1]
        force = torch.clamp(action[..., 0], self.min_action, self.max_action)

        next_velocity = velocity + force * self.power - self.gravity * torch.cos(3.0 * position)
        next_velocity = torch.clamp(next_velocity, -self.max_speed, self.max_speed)
        next_position = torch.clamp(position + next_velocity, self.min_position, self.max_position)
        left_wall = (next_position == self.min_position) & (next_velocity < 0)
        next_velocity = torch.where(left_wall, torch.zeros_like(next_velocity), next_velocity)
        return torch.stack([next_position, next_velocity], dim=-1)

    def interval_step(
        self,
        state_lower: torch.Tensor,
        state_upper: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return conservative next-state bounds for a state/action box."""

        position_l = state_lower[..., 0]
        position_u = state_upper[..., 0]
        velocity_l = state_lower[..., 1]
        velocity_u = state_upper[..., 1]
        action_l, action_u = clip_interval(
            action_lower[..., 0],
            action_upper[..., 0],
            self.min_action,
            self.max_action,
        )

        cos_l, cos_u = cos_interval(3.0 * position_l, 3.0 * position_u)
        velocity_next_l = velocity_l + self.power * action_l - self.gravity * cos_u
        velocity_next_u = velocity_u + self.power * action_u - self.gravity * cos_l
        velocity_next_l, velocity_next_u = clip_interval(
            velocity_next_l,
            velocity_next_u,
            -self.max_speed,
            self.max_speed,
        )

        position_next_l = position_l + velocity_next_l
        position_next_u = position_u + velocity_next_u
        position_next_l, position_next_u = clip_interval(
            position_next_l,
            position_next_u,
            self.min_position,
            self.max_position,
        )

        # The left-wall rule can only increase a negative velocity to zero.
        wall_possible = position_next_l <= self.min_position
        velocity_next_l = torch.where(wall_possible, torch.minimum(velocity_next_l, torch.zeros_like(velocity_next_l)), velocity_next_l)
        velocity_next_u = torch.where(wall_possible, torch.maximum(velocity_next_u, torch.zeros_like(velocity_next_u)), velocity_next_u)

        return (
            torch.stack([position_next_l, velocity_next_l], dim=-1),
            torch.stack([position_next_u, velocity_next_u], dim=-1),
        )
