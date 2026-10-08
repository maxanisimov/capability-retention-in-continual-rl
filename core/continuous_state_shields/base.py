"""Common interface for deterministic discrete-action safety shields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import numpy as np

FallbackRule = Literal["lowest", "highest"] | Callable[[Sequence[int], int], int]


class SafetyShield(ABC):
    """Base class for shields which map one observation to safe actions.

    Unsafe proposed actions are replaced deterministically. The default fallback
    is the numerically lowest safe action. ``fallback_rule="highest"`` or a
    deterministic callable ``(safe_actions, proposed_action) -> action`` can be
    supplied instead.

    Implementations are non-empty by default. When ``allow_empty_safe_set`` is
    true, :meth:`get_safe_actions` may return ``[]`` for a custom subclass, but
    :meth:`shield_action` still raises ``RuntimeError`` because no safe action can
    truthfully be executed.
    """

    def __init__(
        self,
        n_actions: int,
        *,
        fallback_rule: FallbackRule = "lowest",
        allow_empty_safe_set: bool = False,
    ) -> None:
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive; got {n_actions}.")
        if not callable(fallback_rule) and fallback_rule not in ("lowest", "highest"):
            raise ValueError("fallback_rule must be 'lowest', 'highest', or callable.")
        self.n_actions = int(n_actions)
        self.fallback_rule = fallback_rule
        self.allow_empty_safe_set = bool(allow_empty_safe_set)

    @abstractmethod
    def _evaluate(self, state: np.ndarray) -> tuple[list[int], dict[str, Any]]:
        """Return the safe actions and environment-specific diagnostics."""

    @overload
    def get_safe_actions(
        self, state: Any, *, return_info: Literal[False] = False
    ) -> list[int]: ...

    @overload
    def get_safe_actions(
        self, state: Any, *, return_info: Literal[True]
    ) -> tuple[list[int], dict[str, Any]]: ...

    def get_safe_actions(
        self, state: Any, *, return_info: bool = False
    ) -> list[int] | tuple[list[int], dict[str, Any]]:
        """Return sorted admissible actions, optionally with decision diagnostics."""
        observation = np.asarray(state, dtype=float)
        if observation.ndim != 1:
            raise ValueError(
                f"state must be one-dimensional; got shape {observation.shape}."
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError("state must contain only finite values.")

        actions, details = self._evaluate(observation)
        safe_actions = sorted({int(action) for action in actions})
        if any(action < 0 or action >= self.n_actions for action in safe_actions):
            raise RuntimeError(
                f"Shield produced an out-of-range action: {safe_actions}."
            )
        if not safe_actions and not self.allow_empty_safe_set:
            raise RuntimeError("Shield produced an empty safe-action set.")

        info = dict(details)
        info["safe_actions"] = safe_actions.copy()
        info["shield_active"] = len(safe_actions) < self.n_actions
        info.setdefault("violated_constraints", [])
        info.setdefault("reason", "all actions satisfy the configured safety rule")
        return (safe_actions, info) if return_info else safe_actions

    def is_safe_action(self, state: Any, action: int) -> bool:
        """Return whether ``action`` is admissible at ``state``."""
        action = self._validate_action(action)
        return action in self.get_safe_actions(state)

    def shield_action(self, state: Any, proposed_action: int) -> int:
        """Keep a safe action, otherwise apply the configured deterministic fallback."""
        proposed_action = self._validate_action(proposed_action)
        safe_actions = self.get_safe_actions(state)
        if proposed_action in safe_actions:
            return proposed_action
        if not safe_actions:
            raise RuntimeError("No safe action is available for the supplied state.")

        if callable(self.fallback_rule):
            replacement = int(self.fallback_rule(tuple(safe_actions), proposed_action))
            if replacement not in safe_actions:
                raise RuntimeError(
                    f"fallback_rule returned unsafe action {replacement}; safe actions are {safe_actions}."
                )
            return replacement
        return safe_actions[0] if self.fallback_rule == "lowest" else safe_actions[-1]

    def _validate_action(self, action: int) -> int:
        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError(f"action must be an integer; got {action!r}.")
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}); got {action}.")
        return action
