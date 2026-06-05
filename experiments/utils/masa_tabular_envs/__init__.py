"""Custom configurable MASA-style tabular Gymnasium environments."""

from __future__ import annotations

from gymnasium.envs.registration import register

from experiments.utils.masa_tabular_envs.factory import make_custom_masa_env
from experiments.utils.masa_tabular_envs.frozen_lake import CustomFrozenLake
from experiments.utils.masa_tabular_envs.gridworlds import (
    CustomBridgeCrossing,
    CustomBridgeCrossingV2,
    CustomColourBombGridWorld,
    CustomColourBombGridWorldV2,
    CustomColourBombGridWorldV3,
    CustomColourGridWorld,
)
from experiments.utils.masa_tabular_envs.media_streaming import (
    CustomMediaStreaming,
    CustomMediaStreamingV2,
    CustomMediaStreamingV3,
)
from experiments.utils.masa_tabular_envs.pacman import CustomMiniPacman, CustomPacman


def _safe_register(env_id: str, entry_point: str) -> None:
    try:
        register(id=env_id, entry_point=entry_point)
    except Exception:
        pass


_safe_register(
    "CustomFrozenLake-v0",
    "experiments.utils.masa_tabular_envs.frozen_lake:CustomFrozenLake",
)
_safe_register(
    "CustomBridgeCrossing-v0",
    "experiments.utils.masa_tabular_envs.gridworlds:CustomBridgeCrossing",
)
_safe_register(
    "CustomBridgeCrossingV2-v0",
    "experiments.utils.masa_tabular_envs.gridworlds:CustomBridgeCrossingV2",
)
_safe_register(
    "CustomColourGridWorld-v0",
    "experiments.utils.masa_tabular_envs.gridworlds:CustomColourGridWorld",
)
_safe_register(
    "CustomColourBombGridWorld-v0",
    "experiments.utils.masa_tabular_envs.gridworlds:CustomColourBombGridWorld",
)
_safe_register(
    "CustomColourBombGridWorldV2-v0",
    "experiments.utils.masa_tabular_envs.gridworlds:CustomColourBombGridWorldV2",
)
_safe_register(
    "CustomColourBombGridWorldV3-v0",
    "experiments.utils.masa_tabular_envs.gridworlds:CustomColourBombGridWorldV3",
)
_safe_register(
    "CustomMediaStreaming-v0",
    "experiments.utils.masa_tabular_envs.media_streaming:CustomMediaStreaming",
)
_safe_register(
    "CustomMediaStreamingV2-v0",
    "experiments.utils.masa_tabular_envs.media_streaming:CustomMediaStreamingV2",
)
_safe_register(
    "CustomMediaStreamingV3-v0",
    "experiments.utils.masa_tabular_envs.media_streaming:CustomMediaStreamingV3",
)
_safe_register(
    "CustomMiniPacman-v0",
    "experiments.utils.masa_tabular_envs.pacman:CustomMiniPacman",
)
_safe_register(
    "CustomPacman-v0",
    "experiments.utils.masa_tabular_envs.pacman:CustomPacman",
)


__all__ = [
    "CustomFrozenLake",
    "CustomBridgeCrossing",
    "CustomBridgeCrossingV2",
    "CustomColourGridWorld",
    "CustomColourBombGridWorld",
    "CustomColourBombGridWorldV2",
    "CustomColourBombGridWorldV3",
    "CustomMediaStreaming",
    "CustomMediaStreamingV2",
    "CustomMediaStreamingV3",
    "CustomMiniPacman",
    "CustomPacman",
    "make_custom_masa_env",
]
