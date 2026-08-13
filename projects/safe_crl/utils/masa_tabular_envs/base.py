"""Small local equivalent of MASA's tabular environment base class."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ObservationMode = Literal["index", "features"]


class TabularEnv(gym.Env):
    """Gymnasium environment with optional tabular dynamics accessors.

    Observation representation
    --------------------------
    Every state has an integer id used internally for dynamics, the shield and
    value iteration. ``observation_mode`` selects what an *agent* sees:

    * ``"features"`` (default): a normalised ``Box`` vector of the state's
      decoded structured components (grid ``row,col``; media ``buffer,
      fast_count``; the pacman 7-tuple; ...), each divided by its maximum so the
      network has real structure to generalise over instead of an orthogonal
      one-hot.
    * ``"index"``: the raw integer state id (the historical ``Discrete`` view).

    The feature encoding is **exactly invertible** (``features_to_state`` rounds
    each component back to its integer), so the shield -- which is indexed by
    state id -- keeps working via :meth:`make_obs_to_state`. Subclasses supply
    the per-env decode by implementing :meth:`_state_components`,
    :meth:`_components_to_state` and :meth:`_component_maxima`.
    """

    _observation_mode: ObservationMode = "index"
    _feature_maxima: np.ndarray | None = None
    _feature_dim: int | None = None

    def __init__(self) -> None:
        super().__init__()
        self._transition_matrix = None
        self._successor_states = None
        self._transition_probs = None

    def _validate_tabular_spaces(self) -> None:
        if not isinstance(self.observation_space, spaces.Discrete | spaces.Dict | spaces.Box):
            raise TypeError("Tabular env observations must be Discrete, Dict or Box.")
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("Tabular env actions must be Discrete.")

    # ---------------------------------------------------- feature observations

    def _state_components(self, state: int) -> tuple[int, ...]:
        """Decode a state id into its integer structured components."""
        raise NotImplementedError(f"{type(self).__name__} has no feature decode.")

    def _components_to_state(self, components: Sequence[int]) -> int:
        """Re-encode integer components back to a state id (inverse of above)."""
        raise NotImplementedError(f"{type(self).__name__} has no feature encode.")

    def _component_maxima(self) -> tuple[int, ...]:
        """Per-component maximum value, used to normalise features to [0, 1]."""
        raise NotImplementedError(f"{type(self).__name__} has no feature maxima.")

    def _init_observation_mode(self, observation_mode: ObservationMode) -> None:
        """Finalise the observation space for the requested mode.

        Call at the end of ``__init__`` (after the ``Discrete`` space and the
        tabular dynamics are set up). In ``"features"`` mode this replaces the
        observation space with a normalised ``Box``; in ``"index"`` mode it
        leaves the existing ``Discrete`` untouched.
        """
        if observation_mode not in ("index", "features"):
            raise ValueError(
                f"observation_mode must be 'index' or 'features'; got {observation_mode!r}."
            )
        self._observation_mode = observation_mode
        if observation_mode == "features":
            maxima = np.asarray(self._component_maxima(), dtype=np.float64)
            self._feature_maxima = maxima
            self._feature_dim = int(maxima.shape[0])
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self._feature_dim,), dtype=np.float32
            )

    def state_to_features(self, state: int) -> np.ndarray:
        """Normalised float feature vector for a state id."""
        comps = np.asarray(self._state_components(int(state)), dtype=np.float64)
        denom = np.where(self._feature_maxima > 0, self._feature_maxima, 1.0)
        return (comps / denom).astype(np.float32)

    def features_to_state(self, features: Any) -> int:
        """Exact inverse of :meth:`state_to_features`."""
        arr = features.detach().cpu().numpy() if hasattr(features, "detach") else np.asarray(features)
        comps = np.rint(arr.reshape(-1).astype(np.float64) * self._feature_maxima).astype(int)
        return int(self._components_to_state(comps))

    def _observe(self, state: int) -> Any:
        """The observation an agent sees for ``state``, per the current mode."""
        if self._observation_mode == "features":
            return self.state_to_features(state)
        return int(state)

    def make_obs_to_state(self) -> Callable[[Any], np.ndarray]:
        """Batched observation -> state-id map for the shield.

        In ``"index"`` mode this is the identity cast (matching the shield's own
        default); in ``"features"`` mode it un-normalises and rounds each row
        back to a state id. The returned closure holds the decode by reference,
        so it must be used while this env is alive (it is, during training).
        """
        mode = self._observation_mode
        maxima = self._feature_maxima
        components_to_state = self._components_to_state

        def obs_to_state(obs: Any) -> np.ndarray:
            arr = obs.detach().cpu().numpy() if hasattr(obs, "detach") else np.asarray(obs)
            if mode != "features":
                return arr.reshape(-1).astype(np.int64)
            rows = arr.reshape(-1, int(maxima.shape[0])).astype(np.float64)
            comps = np.rint(rows * maxima).astype(int)
            return np.array([components_to_state(c) for c in comps], dtype=np.int64)

        return obs_to_state

    @property
    def has_transition_matrix(self) -> bool:
        return self._transition_matrix is not None

    @property
    def has_successor_states_dict(self) -> bool:
        return self._successor_states is not None and self._transition_probs is not None

    def get_transition_matrix(self):
        return self._transition_matrix

    def get_successor_states_dict(self):
        if not self.has_successor_states_dict:
            return None
        return self._successor_states, self._transition_probs
