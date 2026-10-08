"""RL-library-independent collection of state/safe-action supervision."""

from __future__ import annotations

from typing import Any

import numpy as np

from continuous_state_shields.base import SafetyShield


def collect_shield_sample(state: Any, shield: SafetyShield) -> dict[str, Any]:
    """Copy one state and label it with its complete safe-action set."""
    observation = np.asarray(state).copy()
    return {"state": observation, "safe_actions": shield.get_safe_actions(observation)}


class ShieldDatasetCollector:
    """Minimal in-memory collector for ``(state, safe_actions)`` records."""

    def __init__(self, shield: SafetyShield) -> None:
        self.shield = shield
        self.samples: list[dict[str, Any]] = []

    def append(self, state: Any) -> dict[str, Any]:
        """Append and return a defensive-copy sample."""
        sample = collect_shield_sample(state, self.shield)
        self.samples.append(sample)
        return sample

    def clear(self) -> None:
        self.samples.clear()

    def __len__(self) -> int:
        return len(self.samples)
