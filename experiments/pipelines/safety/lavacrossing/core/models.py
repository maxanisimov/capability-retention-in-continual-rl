"""Actor/critic models for the LavaCrossing shield-safety pipeline."""

from __future__ import annotations

import torch


def _activation_layer(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'. Expected 'relu' or 'tanh'.")


def build_actor_critic(
    *,
    obs_dim: int = 3,
    hidden: int = 64,
    activation: str = "relu",
    n_actions: int = 5,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        _activation_layer(activation),
        torch.nn.Linear(hidden, hidden),
        _activation_layer(activation),
        torch.nn.Linear(hidden, n_actions),
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        _activation_layer(activation),
        torch.nn.Linear(hidden, hidden),
        _activation_layer(activation),
        torch.nn.Linear(hidden, 1),
    )
    return actor, critic
