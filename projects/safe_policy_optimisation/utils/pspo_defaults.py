"""Validated default settings for PSPO experiments.

These values mirror ``docs/pspo_best_settings.yaml``. Keeping the operational
defaults here lets every PSPO entry point share the same
initial-policy and environment-specific training choices while still allowing
explicit command-line or environment-variable overrides.
"""

from __future__ import annotations

from dataclasses import dataclass


BC_INITIALISATION_OBJECTIVE = "margin"
BC_MARGIN_MODE = "all"
BC_TARGET_MARGIN = 2.0
BC_MARGIN_LOSS_WEIGHT = 1.0
BC_SAFE_ACTION_ENTROPY_WEIGHT = 1.0
BC_MIN_SAFE_ACTION_ENTROPY = 0.95
BC_UNSAFE_MASS_TARGET = 0.01
BC_MAX_UNSAFE_MASS = 0.02
BC_SAFE_ACTION_UNIFORMITY_WEIGHT = 1.0

RASHOMON_N_ITERS = 200
RASHOMON_CHECKPOINT = 100
RASHOMON_MULTI_LABEL_MODE = "all"
RASHOMON_SURROGATE = "logsumexp"
RASHOMON_OBJECTIVE = "weighted_width"


@dataclass(frozen=True)
class EnvironmentDefaults:
    total_timesteps: int
    frequency: str


_GENERIC_DEFAULTS = EnvironmentDefaults(total_timesteps=100_000, frequency="1")

_BY_ENVIRONMENT = {
    "media_streaming": EnvironmentDefaults(25_000, "1"),
    "media_streaming_b128": EnvironmentDefaults(50_000, "1"),
    "CustomMediaStreamingV2-v0": EnvironmentDefaults(25_000, "1"),
    "colour_bomb": EnvironmentDefaults(25_000, "1"),
    "CustomColourBombGridWorld-v0": EnvironmentDefaults(25_000, "1"),
    "colour_bomb_v2": EnvironmentDefaults(100_000, "1"),
    "CustomColourBombGridWorldV3-v0": EnvironmentDefaults(100_000, "1"),
    "bridge_crossing": EnvironmentDefaults(200_000, "1"),
    "CustomBridgeCrossing-v0": EnvironmentDefaults(200_000, "1"),
    "bridge_crossing_v2": EnvironmentDefaults(1_600_000, "1"),
    "CustomBridgeCrossingV2-v0": EnvironmentDefaults(1_600_000, "1"),
    "mini_pacman": EnvironmentDefaults(2_000_000, "100"),
    "CustomMiniPacman-v0": EnvironmentDefaults(2_000_000, "100"),
}


def environment_defaults(environment: str | None) -> EnvironmentDefaults:
    """Return best-known defaults, falling back safely for an unknown task."""

    return _BY_ENVIRONMENT.get(str(environment), _GENERIC_DEFAULTS)
