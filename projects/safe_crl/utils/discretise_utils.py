"""
DiscretiseWrapper — Gymnasium wrapper that discretises observations and actions.

Converts every feature in Box observation and Box action spaces into
integer bin indices (or normalised floats), using known min/max bounds
per feature.

By default, bin indices are min-max normalised to [0, 1], so that all
features are on the same scale regardless of how many bins each has.
This can be disabled with normalize=False to get raw integer indices.

    normalised value = bin_index / (n_bins - 1)

Usage
-----
    import gymnasium as gym
    from discretise_wrapper import DiscretiseWrapper

    # Default: normalised to [0, 1]
    env = DiscretiseWrapper(env, obs_bins=..., obs_ranges=...,
                            act_bins=..., act_ranges=...)

    # Raw integer bin indices
    env = DiscretiseWrapper(env, ..., normalize=False)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscretiseWrapper(gym.Wrapper):
    """Discretise Box observations and Box actions.

    Each continuous scalar feature is mapped to a bin index in
    [0, n_bins - 1].  By default, the bin index is then normalised to
    [0, 1] by dividing by (n_bins - 1), so that all features live on
    the same scale.  The resulting observation space is Box([0,1]) when
    normalised or MultiDiscrete when not.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.

    obs_bins : dict[str, int | list[int]] | list[int]
        Number of bins per observation feature.
        - If the observation space is a flat Box of shape (N,), pass a
          list of N integers.
        - If the observation space is a Dict of Box spaces, pass a dict.
          Each value can be a single int (same bins for all features in
          that key) or a list of ints (one per feature).

    obs_ranges : dict[str, tuple | list[tuple]] | list[tuple[float, float]]
        (min, max) range for each observation feature.  Same structure
        as obs_bins.  Raises ValueError if any bound is +/-inf.

    act_bins : list[int]
        Number of bins per action feature.

    act_ranges : list[tuple[float, float]]
        (min, max) range for each action feature.  Raises ValueError
        if any bound is +/-inf.

    normalize : bool
        If True (default), bin indices are divided by (n_bins - 1) to
        produce float values in [0, 1].  If False, raw integer bin
        indices are returned and the space is MultiDiscrete.

    obs_keys : list[str] | None
        If the observation space is a Dict, specifies which keys to
        discretise.  Keys not listed are passed through unchanged.
        If None, all keys are discretised.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_bins: dict[str, int | list[int]] | list[int],
        obs_ranges: dict[str, tuple | list[tuple]] | list[tuple[float, float]],
        act_bins: list[int],
        act_ranges: list[tuple[float, float]],
        normalize: bool = True,
        obs_keys: list[str] | None = None,
    ):
        super().__init__(env)
        self._normalize = normalize

        # ==================================================================
        #  Action config
        # ==================================================================
        inner_act = env.action_space
        if not isinstance(inner_act, spaces.Box):
            raise TypeError(
                f"DiscretiseWrapper requires a Box action space, "
                f"got {type(inner_act).__name__}"
            )
        act_dim = int(np.prod(inner_act.shape))
        if len(act_bins) != act_dim:
            raise ValueError(
                f"act_bins length ({len(act_bins)}) != "
                f"action space dim ({act_dim})"
            )
        if len(act_ranges) != act_dim:
            raise ValueError(
                f"act_ranges length ({len(act_ranges)}) != "
                f"action space dim ({act_dim})"
            )

        self._act_bins = list(act_bins)
        self._act_ranges = list(act_ranges)
        self._act_inner_shape = inner_act.shape

        _validate_ranges(self._act_ranges, "action")

        if normalize:
            self.action_space = spaces.Box(
                0.0, 1.0,
                shape=(act_dim,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.MultiDiscrete(self._act_bins)

        # ==================================================================
        #  Observation config
        # ==================================================================
        inner_obs = env.observation_space
        self._obs_is_dict = isinstance(inner_obs, spaces.Dict)

        if self._obs_is_dict:
            self._obs_keys_to_discretise = (
                obs_keys if obs_keys is not None
                else list(inner_obs.spaces.keys())
            )
            self._obs_passthrough_keys = [
                k for k in inner_obs.spaces.keys()
                if k not in self._obs_keys_to_discretise
            ]

            if not isinstance(obs_bins, dict) or not isinstance(obs_ranges, dict):
                raise TypeError(
                    "For Dict observation spaces, obs_bins and obs_ranges "
                    "must be dicts."
                )

            self._obs_key_configs = {}
            new_spaces = {}

            for key in self._obs_keys_to_discretise:
                box = inner_obs.spaces[key]
                if not isinstance(box, spaces.Box):
                    raise TypeError(
                        f"Observation key '{key}' is {type(box).__name__}, "
                        f"expected Box"
                    )
                dim = int(np.prod(box.shape))

                bins_for_key = obs_bins.get(key)
                ranges_for_key = obs_ranges.get(key)

                if bins_for_key is None or ranges_for_key is None:
                    raise ValueError(
                        f"obs_bins and obs_ranges must contain key '{key}'"
                    )

                # Normalise to lists
                if isinstance(bins_for_key, int):
                    bins_for_key = [bins_for_key] * dim
                if isinstance(ranges_for_key, tuple) and len(ranges_for_key) == 2 and not isinstance(ranges_for_key[0], tuple):
                    ranges_for_key = [ranges_for_key] * dim

                bins_list = list(bins_for_key)
                ranges_list = list(ranges_for_key)

                if len(bins_list) != dim:
                    raise ValueError(
                        f"obs_bins['{key}'] length ({len(bins_list)}) != "
                        f"space dim ({dim})"
                    )
                if len(ranges_list) != dim:
                    raise ValueError(
                        f"obs_ranges['{key}'] length ({len(ranges_list)}) != "
                        f"space dim ({dim})"
                    )
                _validate_ranges(ranges_list, f"observation key '{key}'")

                self._obs_key_configs[key] = (bins_list, ranges_list)

                if normalize:
                    new_spaces[key] = spaces.Box(
                        0.0, 1.0, shape=(dim,), dtype=np.float32,
                    )
                else:
                    new_spaces[key] = spaces.MultiDiscrete(bins_list)

            for key in self._obs_passthrough_keys:
                new_spaces[key] = inner_obs.spaces[key]

            self.observation_space = spaces.Dict(new_spaces)

        else:
            if not isinstance(inner_obs, spaces.Box):
                raise TypeError(
                    f"DiscretiseWrapper requires Box or Dict observation "
                    f"space, got {type(inner_obs).__name__}"
                )
            if not isinstance(obs_bins, list) or not isinstance(obs_ranges, list):
                raise TypeError(
                    "For flat Box observation spaces, obs_bins and "
                    "obs_ranges must be lists."
                )

            obs_dim = int(np.prod(inner_obs.shape))
            if len(obs_bins) != obs_dim:
                raise ValueError(
                    f"obs_bins length ({len(obs_bins)}) != "
                    f"observation space dim ({obs_dim})"
                )
            if len(obs_ranges) != obs_dim:
                raise ValueError(
                    f"obs_ranges length ({len(obs_ranges)}) != "
                    f"observation space dim ({obs_dim})"
                )
            _validate_ranges(obs_ranges, "observation")

            self._obs_bins_flat = list(obs_bins)
            self._obs_ranges_flat = list(obs_ranges)

            if normalize:
                self.observation_space = spaces.Box(
                    0.0, 1.0, shape=(obs_dim,), dtype=np.float32,
                )
            else:
                self.observation_space = spaces.MultiDiscrete(self._obs_bins_flat)

    # ==================================================================
    #  Observation conversion
    # ==================================================================

    def _discretise_obs(self, obs):
        if self._obs_is_dict:
            new_obs = {}
            for key in self._obs_keys_to_discretise:
                bins_list, ranges_list = self._obs_key_configs[key]
                vec = obs[key].flatten()
                new_obs[key] = _discretise_vector(
                    vec, bins_list, ranges_list, self._normalize
                )
            for key in self._obs_passthrough_keys:
                new_obs[key] = obs[key]
            return new_obs
        else:
            vec = obs.flatten()
            return _discretise_vector(
                vec, self._obs_bins_flat, self._obs_ranges_flat, self._normalize
            )

    # ==================================================================
    #  Action conversion
    # ==================================================================

    def _decode_action(self, action):
        """Convert a discretised action back to continuous values."""
        action = np.asarray(action)
        continuous = np.zeros(len(self._act_bins), dtype=np.float32)
        for i, (n_bins, (lo, hi)) in enumerate(
            zip(self._act_bins, self._act_ranges)
        ):
            if self._normalize:
                # Action is in [0, 1] — convert back to bin index
                bin_idx = int(round(float(action[i]) * (n_bins - 1)))
            else:
                bin_idx = int(action[i])
            bin_idx = min(max(bin_idx, 0), n_bins - 1)
            # Map to bin centre
            continuous[i] = lo + (bin_idx + 0.5) * (hi - lo) / n_bins
        return continuous.reshape(self._act_inner_shape)

    # ==================================================================
    #  Gym interface
    # ==================================================================

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._discretise_obs(obs), info

    def step(self, action):
        continuous_action = self._decode_action(action)
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        return self._discretise_obs(obs), reward, terminated, truncated, info


# ======================================================================
#  Helpers
# ======================================================================

def _validate_ranges(ranges, name):
    """Raise ValueError if any bound is infinite."""
    for i, (lo, hi) in enumerate(ranges):
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(
                f"Infinite bound in {name} feature {i}: ({lo}, {hi}). "
                f"All min/max values must be finite."
            )
        if lo >= hi:
            raise ValueError(
                f"Invalid range in {name} feature {i}: lo={lo} >= hi={hi}"
            )


def _discretise_matrix(mat: np.ndarray, bins_list, ranges_list,
                        normalize: bool = False) -> np.ndarray:
    """Vectorised discretisation of a 2-D array.

    Parameters
    ----------
    mat : np.ndarray, shape (N, D)
        Each row is a continuous vector to discretise.
    bins_list : list[int], length D
    ranges_list : list[tuple[float, float]], length D
    normalize : bool
        If True, return normalised float indices in [0, 1].
        If False, return raw integer bin indices.

    Returns
    -------
    np.ndarray, shape (N, D)  float32
    """
    lo   = np.array([r[0] for r in ranges_list], dtype=np.float32)
    hi   = np.array([r[1] for r in ranges_list], dtype=np.float32)
    bins = np.array(bins_list, dtype=np.float32)

    clipped = np.clip(mat, lo, hi)
    indices  = np.floor((clipped - lo) / (hi - lo) * bins).astype(np.int64)
    indices  = np.clip(indices, 0, (bins - 1).astype(np.int64))

    if normalize:
        return (indices / np.maximum(bins - 1, 1)).astype(np.float32)
    return indices.astype(np.float32)


def _bin_index(value, lo, hi, n_bins):
    """Map a continuous value to a bin index in [0, n_bins - 1]."""
    clamped = np.clip(value, lo, hi)
    idx = int((clamped - lo) / (hi - lo) * n_bins)
    return min(max(idx, 0), n_bins - 1)


def _discretise_vector(vec, bins_list, ranges_list, normalize):
    """Discretise a flat numpy vector into bin indices or normalised floats."""
    if normalize:
        result = np.zeros(len(bins_list), dtype=np.float32)
        for i, (n_bins, (lo, hi)) in enumerate(zip(bins_list, ranges_list)):
            idx = _bin_index(float(vec[i]), lo, hi, n_bins)
            result[i] = idx / (n_bins - 1) if n_bins > 1 else 0.0
    else:
        result = np.zeros(len(bins_list), dtype=np.int64)
        for i, (n_bins, (lo, hi)) in enumerate(zip(bins_list, ranges_list)):
            result[i] = _bin_index(float(vec[i]), lo, hi, n_bins)
    return result
