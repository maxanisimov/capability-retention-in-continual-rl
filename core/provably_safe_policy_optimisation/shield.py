"""Runtime safety shield for discrete state-action spaces.

A shield is a binary mask ``(n_states, n_actions)`` where ``mask[s, a] == 1`` means
action ``a`` is *safe* in state ``s`` (the representation produced by
``experiments.utils.shield_utils.synthesise_shield`` and stored in ``shield_q.pt``
artifacts). At runtime the shield maps an observation to a discrete state, and if a
chosen action is unsafe it overrides it by sampling **uniformly from that state's safe
actions**.

This module is independent of how the shield was synthesised; users supply a finished
mask (hand-built, from ``synthesise_shield``, or loaded from disk).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal

import numpy as np

# obs (batch) -> integer state ids, shape (n_envs,)
ObsToState = Callable[[Any], np.ndarray]


def _default_obs_to_state(obs: Any) -> np.ndarray:
    """Treat a (batch of) discrete observation(s) as the state id(s).

    Accepts numpy arrays or torch tensors. Suitable for ``Discrete`` observation
    spaces; non-trivial encodings require a custom ``obs_to_state``.
    """
    if hasattr(obs, "detach"):  # torch.Tensor
        obs = obs.detach().cpu().numpy()
    arr = np.asarray(obs)
    return arr.reshape(arr.shape[0] if arr.ndim else 1, -1).squeeze(-1).astype(np.int64).reshape(-1)


class Shield:
    """A discrete state-action safety shield.

    Parameters
    ----------
    mask:
        Array-like ``(n_states, n_actions)``; ``mask[s, a]`` truthy means action
        ``a`` is safe in state ``s``.
    obs_to_state:
        Maps a batch of observations to integer state ids. Defaults to
        :func:`_default_obs_to_state` (the observation *is* the state id).
    no_safe_action:
        Behaviour when a state has no safe action: ``"keep"`` (leave the proposed
        action unchanged, count it, warn once) or ``"raise"``.
    seed:
        Seed for the fallback RNG used by :meth:`override` when no explicit RNG is
        passed.
    """

    def __init__(
        self,
        mask: Any,
        obs_to_state: ObsToState | None = None,
        *,
        no_safe_action: Literal["keep", "raise"] = "keep",
        seed: int | None = None,
    ) -> None:
        if hasattr(mask, "detach"):  # torch.Tensor
            mask = mask.detach().cpu().numpy()
        mask_arr = np.asarray(mask)
        if mask_arr.ndim != 2:
            raise ValueError(f"Shield mask must be 2-D (n_states, n_actions); got shape {mask_arr.shape}.")
        self._mask = (mask_arr != 0)
        if no_safe_action not in ("keep", "raise"):
            raise ValueError(f"no_safe_action must be 'keep' or 'raise'; got {no_safe_action!r}.")
        self._no_safe_action = no_safe_action
        self.obs_to_state: ObsToState = obs_to_state or _default_obs_to_state
        self._rng = np.random.default_rng(seed)
        self._warned_no_safe = False
        self.reset_diagnostics()

    @property
    def n_states(self) -> int:
        return int(self._mask.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self._mask.shape[1])

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    def safe_actions(self, state: int) -> np.ndarray:
        """Indices of safe actions in ``state``."""
        return np.flatnonzero(self._mask[int(state)])

    def is_safe(self, state: int, action: int) -> bool:
        return bool(self._mask[int(state), int(action)])

    def override(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return safe actions: keep each safe action, else resample uniformly safe.

        ``states`` and ``actions`` are 1-D arrays of equal length (one entry per
        environment). Updates the diagnostics counters.
        """
        rng = rng if rng is not None else self._rng
        states = np.asarray(states).astype(np.int64).reshape(-1)
        actions = np.asarray(actions).astype(np.int64).reshape(-1)
        if states.shape != actions.shape:
            raise ValueError(f"states and actions must have the same shape; got {states.shape} vs {actions.shape}.")

        out = actions.copy()
        for i, (s, a) in enumerate(zip(states, actions)):
            self._n_checked += 1
            if self._mask[s, a]:
                continue
            safe = np.flatnonzero(self._mask[s])
            if safe.size == 0:
                self._n_no_safe_state += 1
                if self._no_safe_action == "raise":
                    raise RuntimeError(f"Shield: state {int(s)} has no safe action.")
                if not self._warned_no_safe:
                    warnings.warn(
                        f"Shield: state {int(s)} has no safe action; keeping the proposed "
                        "action. (Further occurrences are silenced.)",
                        stacklevel=2,
                    )
                    self._warned_no_safe = True
                continue
            out[i] = int(safe[rng.integers(safe.size)])
            self._n_overridden += 1
        return out

    # --- diagnostics -------------------------------------------------------
    def reset_diagnostics(self) -> None:
        self._n_checked = 0
        self._n_overridden = 0
        self._n_no_safe_state = 0

    @property
    def intervention_rate(self) -> float:
        return (self._n_overridden / self._n_checked) if self._n_checked else 0.0

    def diagnostics(self) -> dict[str, float]:
        return {
            "checked": int(self._n_checked),
            "overridden": int(self._n_overridden),
            "no_safe_state": int(self._n_no_safe_state),
            "intervention_rate": self.intervention_rate,
        }


def as_shield(shield: Any, obs_to_state: ObsToState | None = None, *, seed: int | None = None) -> Shield:
    """Coerce a ``Shield`` or a raw ``(n_states, n_actions)`` mask into a ``Shield``."""
    if isinstance(shield, Shield):
        if obs_to_state is not None:
            shield.obs_to_state = obs_to_state
        return shield
    return Shield(shield, obs_to_state, seed=seed)
