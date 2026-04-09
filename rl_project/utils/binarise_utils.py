"""
BinariseWrapper — Gymnasium wrapper that binarises observations and actions.

Converts every feature in Box observation and Box action spaces into a
fixed-width binary encoding, using known min/max bounds per feature.

Binary encoding preserves ordinal structure efficiently: the MSB splits
the range in half, the next bit splits each half, etc.  This gives
O(log2 N) bits per feature instead of O(N) for one-hot.

Usage
-----
    import gymnasium as gym
    from binarise_wrapper import BinariseWrapper

    env = gym.make("parking-v0")
    wrapped = BinariseWrapper(
        env,
        obs_bits={"x": 8, "y": 8, "vx": 6, "vy": 6, "cos_h": 6, "sin_h": 6},
        obs_ranges={"x": (-50, 50), "y": (-50, 50), "vx": (-10, 10),
                     "vy": (-10, 10), "cos_h": (-1, 1), "sin_h": (-1, 1)},
        act_bits=[5, 5],
        act_ranges=[(-1, 1), (-1, 1)],
    )
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BinariseWrapper(gym.Wrapper):
    """Binarise Box observations and Box actions using known min/max bounds.

    Each continuous scalar feature is discretised into 2^B bins (where B is
    the number of bits assigned to that feature), and the bin index is
    represented as a binary vector of length B.  The resulting observation
    and action spaces are MultiBinary.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.  Must have Box observation_space and
        Box action_space (or Dict observation_space whose leaf values
        are Box — see obs_keys).

    obs_bits : dict[str, int] | list[int]
        Number of binary bits per observation feature.
        - If the observation space is a flat Box of shape (N,), pass a
          list of N integers.
        - If the observation space is a Dict of Box spaces, pass a dict
          mapping feature names to bit counts.  Every leaf Box must be
          1-D and the list length must match the Box dimension.

    obs_ranges : dict[str, tuple[float, float]] | list[tuple[float, float]]
        (min, max) range for each observation feature.  Same structure as
        obs_bits.  Raises ValueError if any bound is +/-inf.

    act_bits : list[int]
        Number of binary bits per action feature.  Length must match the
        action space dimension.

    act_ranges : list[tuple[float, float]]
        (min, max) range for each action feature.  Raises ValueError if
        any bound is +/-inf.

    obs_keys : list[str] | None
        If the observation space is a Dict, specifies which keys contain
        Box spaces that should be binarised.  Keys not listed are passed
        through unchanged.  If None and the obs space is a Dict, ALL keys
        are binarised.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_bits: dict[str, int] | list[int],
        obs_ranges: dict[str, tuple[float, float]] | list[tuple[float, float]],
        act_bits: list[int],
        act_ranges: list[tuple[float, float]],
        obs_keys: list[str] | None = None,
    ):
        super().__init__(env)

        # ==================================================================
        #  Validate and store action config
        # ==================================================================
        inner_act = env.action_space
        if not isinstance(inner_act, spaces.Box):
            raise TypeError(
                f"BinariseWrapper requires a Box action space, "
                f"got {type(inner_act).__name__}"
            )
        act_dim = int(np.prod(inner_act.shape))
        if len(act_bits) != act_dim:
            raise ValueError(
                f"act_bits length ({len(act_bits)}) != "
                f"action space dim ({act_dim})"
            )
        if len(act_ranges) != act_dim:
            raise ValueError(
                f"act_ranges length ({len(act_ranges)}) != "
                f"action space dim ({act_dim})"
            )

        self._act_bits = list(act_bits)
        self._act_ranges = list(act_ranges)
        self._act_total_bits = sum(self._act_bits)
        self._act_inner_shape = inner_act.shape

        _validate_ranges(self._act_ranges, "action")

        # Build the discrete action grid: enumerate all 2^total_bits combos
        # is only feasible for small total bits.  For large spaces we just
        # decode the binary action on the fly.
        self._act_n_bins = [1 << b for b in self._act_bits]

        self.action_space = spaces.MultiBinary(self._act_total_bits)

        # ==================================================================
        #  Validate and store observation config
        # ==================================================================
        inner_obs = env.observation_space
        self._obs_is_dict = isinstance(inner_obs, spaces.Dict)

        if self._obs_is_dict:
            self._obs_keys_to_binarise = (
                obs_keys if obs_keys is not None else list(inner_obs.spaces.keys())
            )
            self._obs_passthrough_keys = [
                k for k in inner_obs.spaces.keys()
                if k not in self._obs_keys_to_binarise
            ]

            if not isinstance(obs_bits, dict) or not isinstance(obs_ranges, dict):
                raise TypeError(
                    "For Dict observation spaces, obs_bits and obs_ranges "
                    "must be dicts mapping feature names to values."
                )

            # Per-key config
            self._obs_key_configs = {}  # key -> (bits_list, ranges_list, total_bits)
            new_spaces = {}

            for key in self._obs_keys_to_binarise:
                box = inner_obs.spaces[key]
                if not isinstance(box, spaces.Box):
                    raise TypeError(
                        f"Observation key '{key}' is {type(box).__name__}, "
                        f"expected Box"
                    )
                dim = int(np.prod(box.shape))

                # obs_bits/obs_ranges can be:
                #  - dict with key -> single int/tuple (if dim == 1)
                #  - dict with key -> list of int/tuple (if dim > 1)
                bits_for_key = obs_bits.get(key)
                ranges_for_key = obs_ranges.get(key)

                if bits_for_key is None or ranges_for_key is None:
                    raise ValueError(
                        f"obs_bits and obs_ranges must contain key '{key}'"
                    )

                # Normalise to lists
                if isinstance(bits_for_key, int):
                    bits_for_key = [bits_for_key] * dim
                if isinstance(ranges_for_key, tuple) and len(ranges_for_key) == 2 and not isinstance(ranges_for_key[0], tuple):
                    ranges_for_key = [ranges_for_key] * dim

                bits_list = list(bits_for_key)
                ranges_list = list(ranges_for_key)

                if len(bits_list) != dim:
                    raise ValueError(
                        f"obs_bits['{key}'] length ({len(bits_list)}) != "
                        f"space dim ({dim})"
                    )
                if len(ranges_list) != dim:
                    raise ValueError(
                        f"obs_ranges['{key}'] length ({len(ranges_list)}) != "
                        f"space dim ({dim})"
                    )
                _validate_ranges(ranges_list, f"observation key '{key}'")

                total = sum(bits_list)
                self._obs_key_configs[key] = (bits_list, ranges_list, total)
                new_spaces[key] = spaces.MultiBinary(total)

            for key in self._obs_passthrough_keys:
                new_spaces[key] = inner_obs.spaces[key]

            self.observation_space = spaces.Dict(new_spaces)

        else:
            # Flat Box observation
            if not isinstance(inner_obs, spaces.Box):
                raise TypeError(
                    f"BinariseWrapper requires Box or Dict observation space, "
                    f"got {type(inner_obs).__name__}"
                )

            if not isinstance(obs_bits, list) or not isinstance(obs_ranges, list):
                raise TypeError(
                    "For flat Box observation spaces, obs_bits and "
                    "obs_ranges must be lists."
                )

            obs_dim = int(np.prod(inner_obs.shape))
            if len(obs_bits) != obs_dim:
                raise ValueError(
                    f"obs_bits length ({len(obs_bits)}) != "
                    f"observation space dim ({obs_dim})"
                )
            if len(obs_ranges) != obs_dim:
                raise ValueError(
                    f"obs_ranges length ({len(obs_ranges)}) != "
                    f"observation space dim ({obs_dim})"
                )
            _validate_ranges(obs_ranges, "observation")

            self._obs_bits_flat = list(obs_bits)
            self._obs_ranges_flat = list(obs_ranges)
            self._obs_total_bits = sum(self._obs_bits_flat)

            self.observation_space = spaces.MultiBinary(self._obs_total_bits)

    # ==================================================================
    #  Observation conversion
    # ==================================================================

    def _binarise_obs(self, obs):
        if self._obs_is_dict:
            new_obs = {}
            for key in self._obs_keys_to_binarise:
                bits_list, ranges_list, _ = self._obs_key_configs[key]
                vec = obs[key].flatten()
                new_obs[key] = _binarise_vector(vec, bits_list, ranges_list)
            for key in self._obs_passthrough_keys:
                new_obs[key] = obs[key]
            return new_obs
        else:
            vec = obs.flatten()
            return _binarise_vector(vec, self._obs_bits_flat, self._obs_ranges_flat)

    # ==================================================================
    #  Action conversion
    # ==================================================================

    def _decode_action(self, binary_action):
        """Convert a binary action vector back to continuous values."""
        binary_action = np.asarray(binary_action, dtype=np.int8)
        continuous = np.zeros(len(self._act_bits), dtype=np.float32)
        offset = 0
        for i, (n_bits, (lo, hi)) in enumerate(
            zip(self._act_bits, self._act_ranges)
        ):
            bits = binary_action[offset : offset + n_bits]
            bin_idx = _binary_array_to_int(bits)
            n_bins = self._act_n_bins[i]
            # Map bin index back to continuous value (bin centre)
            continuous[i] = lo + (bin_idx + 0.5) * (hi - lo) / n_bins
            offset += n_bits
        return continuous.reshape(self._act_inner_shape)

    # ==================================================================
    #  Gym interface
    # ==================================================================

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._binarise_obs(obs), info

    def step(self, action):
        continuous_action = self._decode_action(action)
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        return self._binarise_obs(obs), reward, terminated, truncated, info


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


def _bin_index(value, lo, hi, n_bins):
    """Map a continuous value to a bin index in [0, n_bins - 1]."""
    clamped = np.clip(value, lo, hi)
    idx = int((clamped - lo) / (hi - lo) * n_bins)
    return min(max(idx, 0), n_bins - 1)


def _int_to_binary_array(val, n_bits):
    """Convert an integer to a binary array of length n_bits (MSB first)."""
    bits = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits):
        bits[n_bits - 1 - i] = val & 1
        val >>= 1
    return bits


def _binary_array_to_int(bits):
    """Convert a binary array (MSB first) back to an integer."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def _binarise_vector(vec, bits_list, ranges_list):
    """Binarise a flat numpy vector given per-feature bits and ranges."""
    parts = []
    for i, (n_bits, (lo, hi)) in enumerate(zip(bits_list, ranges_list)):
        n_bins = 1 << n_bits
        idx = _bin_index(float(vec[i]), lo, hi, n_bins)
        parts.append(_int_to_binary_array(idx, n_bits))
    return np.concatenate(parts)