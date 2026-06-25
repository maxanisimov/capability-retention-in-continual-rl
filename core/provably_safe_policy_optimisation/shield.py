"""Runtime safety shields for discrete actions.

A shield maps an observation to an integer index and overrides unsafe actions using a
binary mask ``(n_index, n_actions)`` where ``mask[i, a] == 1`` means action ``a`` is
*safe* at index ``i``. If a chosen action is unsafe it is replaced by an action sampled
**uniformly from that index's safe actions**.

Two flavours, sharing the same override engine and diagnostics:

* :class:`Shield` -- discrete states: index = state id (the representation produced by
  ``experiments.utils.shield_utils.synthesise_shield`` and stored in ``shield_q.pt``).
* :class:`RegionShield` -- continuous states: index = region id, where the continuous
  observation space is split into disjoint regions, each with a fixed safe-action set
  (e.g. MountainCar: "position < 1.0 -> only push right is safe").

This module is independent of how a shield was synthesised; users supply a finished mask
(``Shield``) or a set of regions (``RegionShield``).
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


class RegionShield(Shield):
    """A continuous-state shield: disjoint regions, each with a fixed safe-action set.

    The continuous observation space is partitioned into ordered regions. A region is a
    predicate over a single observation vector; the first region whose predicate holds is
    the observation's index. Observations matching no region fall into a trailing
    *fallback* index (all actions safe by default). Internally this is a :class:`Shield`
    whose ``obs_to_state`` is the region classifier and whose mask is indexed by region,
    so it works unchanged with ``ProvablySafeDQN``/``ProvablySafePPO``.

    Parameters
    ----------
    regions:
        Ordered list of ``(condition, safe_actions)``. ``condition`` is a callable taking
        a single observation vector and returning a bool; ``safe_actions`` is an iterable
        of action indices. Regions are assumed disjoint; ties resolve first-match-wins.
    n_actions:
        Size of the discrete action space.
    default_safe_actions:
        Safe actions for observations matching no region. ``None`` (default) means *all*
        actions are safe (unconstrained).
    no_safe_action, seed:
        Forwarded to :class:`Shield`.
    """

    def __init__(
        self,
        regions: list[tuple[Callable[[np.ndarray], Any], Any]],
        n_actions: int,
        *,
        default_safe_actions: Any = None,
        no_safe_action: Literal["keep", "raise"] = "keep",
        seed: int | None = None,
    ) -> None:
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive; got {n_actions}.")
        conditions: list[Callable[[np.ndarray], Any]] = []
        # mask rows: one per region + a trailing fallback row.
        mask = np.zeros((len(regions) + 1, n_actions), dtype=int)
        for i, region in enumerate(regions):
            condition, safe_actions = region
            if not callable(condition):
                raise ValueError(f"Region {i}: condition must be callable.")
            conditions.append(condition)
            mask[i, self._validate_actions(safe_actions, n_actions, i)] = 1
        fallback = (
            np.arange(n_actions)
            if default_safe_actions is None
            else self._validate_actions(default_safe_actions, n_actions, "fallback")
        )
        mask[len(regions), fallback] = 1

        self._conditions = conditions
        self._fallback_index = len(regions)
        # Axis-aligned box bounds per region (set by from_boxes); enables IBP
        # certification. None for predicate-based regions (empirical checks only).
        self._boxes: list[tuple[np.ndarray, np.ndarray]] | None = None
        super().__init__(mask, obs_to_state=self._classify, no_safe_action=no_safe_action, seed=seed)

    @property
    def has_boxes(self) -> bool:
        """Whether regions have axis-aligned box bounds (required for IBP certification)."""
        return self._boxes is not None

    @property
    def boxes(self) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Per-region ``(low, high)`` bounds, or ``None`` for predicate-based regions."""
        return self._boxes

    @staticmethod
    def _validate_actions(actions: Any, n_actions: int, where: Any) -> np.ndarray:
        arr = np.asarray(list(actions), dtype=np.int64).reshape(-1)
        if arr.size and (arr.min() < 0 or arr.max() >= n_actions):
            raise ValueError(f"Region {where}: safe_actions must be in [0, {n_actions}); got {arr.tolist()}.")
        return arr

    def _classify(self, obs: Any) -> np.ndarray:
        if hasattr(obs, "detach"):  # torch.Tensor
            obs = obs.detach().cpu().numpy()
        batch = np.asarray(obs, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        region_ids = np.full(batch.shape[0], self._fallback_index, dtype=np.int64)
        assigned = np.zeros(batch.shape[0], dtype=bool)
        for idx, condition in enumerate(self._conditions):
            for row in np.flatnonzero(~assigned):
                if bool(condition(batch[row])):
                    region_ids[row] = idx
                    assigned[row] = True
            if assigned.all():
                break
        return region_ids

    @classmethod
    def from_boxes(
        cls,
        boxes: list[tuple[Any, Any, Any]],
        n_actions: int,
        **kwargs: Any,
    ) -> "RegionShield":
        """Build a ``RegionShield`` from axis-aligned boxes.

        ``boxes`` is a list of ``(lows, highs, safe_actions)`` where ``lows``/``highs`` are
        per-dimension bounds (use ``-np.inf``/``np.inf`` for unconstrained dimensions). A
        region matches when ``lows <= obs <= highs`` element-wise (closed intervals).
        """
        regions: list[tuple[Callable[[np.ndarray], Any], Any]] = []
        box_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for i, (lows, highs, safe_actions) in enumerate(boxes):
            low = np.asarray(lows, dtype=np.float64).reshape(-1)
            high = np.asarray(highs, dtype=np.float64).reshape(-1)
            if low.shape != high.shape:
                raise ValueError(f"Box {i}: lows and highs must have the same shape; got {low.shape} vs {high.shape}.")
            if np.any(low > high):
                raise ValueError(f"Box {i}: lows must be <= highs in every dimension.")

            def condition(obs: np.ndarray, low=low, high=high) -> bool:
                return bool(np.all((obs >= low) & (obs <= high)))

            regions.append((condition, safe_actions))
            box_bounds.append((low, high))
        shield = cls(regions, n_actions, **kwargs)
        shield._boxes = box_bounds
        return shield


def as_shield(shield: Any, obs_to_state: ObsToState | None = None, *, seed: int | None = None) -> Shield:
    """Coerce a ``Shield`` or a raw ``(n_states, n_actions)`` mask into a ``Shield``."""
    if isinstance(shield, Shield):
        if obs_to_state is not None:
            shield.obs_to_state = obs_to_state
        return shield
    return Shield(shield, obs_to_state, seed=seed)
