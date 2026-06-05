"""Unit tests for tabular shield synthesis utilities."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    from experiments.utils.shield_utils import synthesise_shield
except ModuleNotFoundError:  # pragma: no cover - exercised only without RL extras
    gym = None


@unittest.skipIf(gym is None, "gymnasium is not installed")
class ShieldUtilsTests(unittest.TestCase):
    def test_deterministic_shield_allows_only_almost_sure_safe_actions(self) -> None:
        env = _TinyShieldEnv()

        shield = synthesise_shield(
            env,
            _transition_matrix,
            _label_fn,
            _cost_fn,
            use_masa_helper=False,
        )

        expected = np.array(
            [
                [1, 0, 1],
                [1, 1, 1],
                [0, 0, 0],
            ],
            dtype=int,
        )
        self.assertTrue(np.array_equal(shield, expected))

    def test_probabilistic_shield_thresholds_eventual_unsafe_risk(self) -> None:
        env = _TinyShieldEnv()

        shield, info = synthesise_shield(
            env,
            _transition_matrix,
            _label_fn,
            _cost_fn,
            shield_type="probabilistic",
            risk_threshold=0.25,
            use_masa_helper=False,
            return_info=True,
        )

        expected = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
            ],
            dtype=int,
        )
        self.assertTrue(np.array_equal(shield, expected))
        self.assertAlmostEqual(float(info.action_risk[0, 1]), 0.2)
        self.assertAlmostEqual(float(info.action_risk[2, 0]), 1.0)


if gym is not None:

    class _TinyShieldEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = spaces.Discrete(3)
            self.action_space = spaces.Discrete(3)
else:

    class _TinyShieldEnv:
        pass


def _transition_matrix(env: gym.Env) -> np.ndarray:
    del env
    matrix = np.zeros((3, 3, 3), dtype=np.float64)
    matrix[1, 0, 0] = 1.0
    matrix[1, 0, 1] = 0.8
    matrix[2, 0, 1] = 0.2
    matrix[0, 0, 2] = 1.0
    matrix[1, 1, :] = 1.0
    matrix[2, 2, :] = 1.0
    return matrix


def _label_fn(obs: int) -> set[str]:
    return {"unsafe"} if int(obs) == 2 else set()


def _cost_fn(labels: set[str]) -> float:
    return 1.0 if "unsafe" in labels else 0.0


if __name__ == "__main__":
    unittest.main()
