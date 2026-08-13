"""PyTorch policy implementations for verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from barrier_tools.policies.interval_bounds import bound_sequential
from barrier_tools.verification.interval import clip_interval


class ConstantPolicy:
    """A deterministic constant-action policy."""

    def __init__(self, action: float) -> None:
        self.action = float(action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.full((*state.shape[:-1], 1), self.action, dtype=state.dtype, device=state.device)

    def interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del upper
        action = self.forward(lower)
        return action, action


class FeedForwardPolicy(nn.Module):
    """Small feed-forward policy with optional normalization and action clipping."""

    def __init__(
        self,
        model: nn.Sequential,
        *,
        obs_mean: torch.Tensor | None = None,
        obs_scale: torch.Tensor | None = None,
        action_low: float = -1.0,
        action_high: float = 1.0,
        clip_action: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("obs_mean", torch.zeros(2) if obs_mean is None else obs_mean.float())
        self.register_buffer("obs_scale", torch.ones(2) if obs_scale is None else obs_scale.float())
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.clip_action = bool(clip_action)

    def _normalize_interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.obs_scale.to(dtype=lower.dtype, device=lower.device)
        mean = self.obs_mean.to(dtype=lower.dtype, device=lower.device)
        norm_l = (lower - mean) / scale
        norm_u = (upper - mean) / scale
        return torch.minimum(norm_l, norm_u), torch.maximum(norm_l, norm_u)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        scale = self.obs_scale.to(dtype=state.dtype, device=state.device)
        mean = self.obs_mean.to(dtype=state.dtype, device=state.device)
        action = self.model((state - mean) / scale)
        if self.clip_action:
            action = torch.clamp(action, self.action_low, self.action_high)
        return action

    def interval(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_l, norm_u = self._normalize_interval(lower, upper)
        action_l, action_u = bound_sequential(self.model, norm_l, norm_u)
        if self.clip_action:
            action_l, action_u = clip_interval(action_l, action_u, self.action_low, self.action_high)
        return action_l, action_u

    def checkpoint_payload(self, architecture: list[int]) -> dict[str, Any]:
        """Return a portable checkpoint payload for this policy."""

        return {
            "architecture": architecture,
            "state_dict": self.state_dict(),
            "obs_mean": self.obs_mean.detach().cpu(),
            "obs_scale": self.obs_scale.detach().cpu(),
            "action_low": self.action_low,
            "action_high": self.action_high,
            "clip_action": self.clip_action,
        }


def build_mlp(architecture: list[int], *, activation: str = "tanh", final_tanh: bool = True) -> nn.Sequential:
    """Build a supported MLP from layer sizes, e.g. ``[2, 16, 16, 1]``."""

    if len(architecture) < 2:
        raise ValueError("Policy architecture must contain at least input and output sizes.")
    if activation not in {"tanh", "relu"}:
        raise ValueError("Supported activations are 'tanh' and 'relu'.")
    activation_cls: type[nn.Module] = nn.Tanh if activation == "tanh" else nn.ReLU
    layers: list[nn.Module] = []
    for idx, (in_features, out_features) in enumerate(zip(architecture[:-1], architecture[1:])):
        layers.append(nn.Linear(int(in_features), int(out_features)))
        is_last = idx == len(architecture) - 2
        if not is_last:
            layers.append(activation_cls())
        elif final_tanh:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def make_default_policy(seed: int = 0) -> FeedForwardPolicy:
    """Create a deterministic small policy used when no checkpoint is supplied."""

    torch.manual_seed(int(seed))
    model = build_mlp([2, 16, 16, 1], activation="tanh", final_tanh=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(0.1)
    return FeedForwardPolicy(model)


def load_policy(config: dict[str, Any], *, device: str = "cpu") -> ConstantPolicy | FeedForwardPolicy:
    """Load a policy from a YAML-derived configuration dictionary."""

    kind = str(config.get("type", "generated"))
    if kind == "constant":
        return ConstantPolicy(float(config["action"]))
    if kind == "generated":
        return make_default_policy(seed=int(config.get("seed", 0))).to(device)
    if kind != "pytorch":
        raise ValueError(f"Unsupported policy type: {kind}")

    architecture = [int(v) for v in config["architecture"]]
    model = build_mlp(
        architecture,
        activation=str(config.get("activation", "tanh")),
        final_tanh=bool(config.get("final_tanh", True)),
    )
    checkpoint = torch.load(Path(config["checkpoint"]), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    obs_mean = torch.as_tensor(config.get("obs_mean", checkpoint.get("obs_mean", [0.0, 0.0])))
    obs_scale = torch.as_tensor(config.get("obs_scale", checkpoint.get("obs_scale", [1.0, 1.0])))
    return FeedForwardPolicy(
        model.to(device),
        obs_mean=obs_mean,
        obs_scale=obs_scale,
        action_low=float(config.get("action_low", checkpoint.get("action_low", -1.0))),
        action_high=float(config.get("action_high", checkpoint.get("action_high", 1.0))),
        clip_action=bool(config.get("clip_action", checkpoint.get("clip_action", True))),
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
