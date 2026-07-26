"""Tests for the env-shielded projected-PPO stage helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from projects.safe_policy_optimisation.stages.train_env_shielded_projected_policy import (
    DEFAULT_OUTPUT_DIR,
    SingleProposalShieldAdapter,
    UnshieldedCostInfoWrapper,
    build_parser,
    synthesise_safest_action_mask,
)


class _StubShieldedEnv(gym.Env):
    """Mimics the ProbShieldWrapperDisc interface seen by the adapter."""

    def __init__(self, n_states: int = 4, n_actions: int = 5, granularity: int = 20) -> None:
        self.observation_space = spaces.Dict(
            {
                "orig_obs": spaces.Discrete(n_states),
                "safety_bound": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([n_actions, n_actions, granularity + 1])
        self.received_actions: list[np.ndarray] = []
        self.label_fn = lambda state: {"lava"} if state == 3 else set()
        self.cost_fn = lambda labels: 1.0 if "lava" in labels else 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return {"orig_obs": 0, "safety_bound": 0.2}, {}

    def step(self, action):
        self.received_actions.append(np.asarray(action))
        return {"orig_obs": 3, "safety_bound": 0.1}, 1.0, False, False, {}


class _StubTabularEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(2)
        self.label_fn = lambda state: {"lava"} if state == 3 else set()
        self.cost_fn = lambda labels: 1.0 if "lava" in labels else 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 3, 0.0, True, False, {}


class SingleProposalShieldAdapterTests(unittest.TestCase):
    def test_exposes_original_spaces(self) -> None:
        adapter = SingleProposalShieldAdapter(_StubShieldedEnv())

        self.assertEqual(adapter.observation_space, spaces.Discrete(4))
        self.assertEqual(adapter.action_space, spaces.Discrete(5))

    def test_step_forwards_edge_mode_proposal_and_hides_shield_state(self) -> None:
        stub = _StubShieldedEnv()
        adapter = SingleProposalShieldAdapter(stub)

        obs, reward, terminated, truncated, info = adapter.step(2)

        self.assertTrue(np.array_equal(stub.received_actions[0], np.array([2, 2, 0])))
        self.assertEqual(int(obs), 3)
        self.assertEqual(reward, 1.0)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["cost"], 1.0)
        self.assertTrue(info["violated_step"])
        self.assertEqual(info["orig_obs"], 3)
        self.assertAlmostEqual(info["safety_bound"], 0.1)

    def test_reset_strips_dict_observation(self) -> None:
        adapter = SingleProposalShieldAdapter(_StubShieldedEnv())

        obs, _info = adapter.reset()

        self.assertEqual(int(obs), 0)

    def test_rejects_env_without_augmented_spaces(self) -> None:
        with self.assertRaises(TypeError):
            SingleProposalShieldAdapter(_StubTabularEnv())


class UnshieldedCostInfoWrapperTests(unittest.TestCase):
    def test_step_reports_label_cost(self) -> None:
        env = UnshieldedCostInfoWrapper(_StubTabularEnv())

        _obs, _reward, _terminated, _truncated, info = env.step(0)

        self.assertEqual(info["cost"], 1.0)
        self.assertTrue(info["violated_step"])


class SafestActionMaskTests(unittest.TestCase):
    def test_media_streaming_vi_reports_safe_set_stats(self) -> None:
        mask, stats = synthesise_safest_action_mask(
            "CustomMediaStreaming-v0",
            max_episode_steps=10,
            env_kwargs={"fast_rate": 0.0, "slow_rate": 0.0, "out_rate": 0.0},
            theta=1e-10,
            max_vi_steps=1000,
            unsafe_cost_threshold=0.5,
            seed=0,
        )

        self.assertEqual(mask.shape, (stats["n_states"], stats["n_actions"]))
        self.assertGreater(stats["states_with_safe_action"], 0)
        self.assertGreaterEqual(stats["mean_safe_set_size"], 1.0)
        self.assertLessEqual(stats["max_safe_set_size"], stats["n_actions"])
        # Every state with a safe action allows only risk-minimising actions.
        row_sums = mask.sum(axis=1)
        self.assertTrue((row_sums[row_sums > 0] >= 1).all())


class CliParsingTests(unittest.TestCase):
    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args(["--env-id", "CustomBridgeCrossing-v0"])

        self.assertEqual(args.env_id, "CustomBridgeCrossing-v0")
        self.assertEqual(args.safety_tolerance, 0.0)
        self.assertEqual(args.theta, 1e-10)
        self.assertEqual(args.max_vi_steps, 1000)
        self.assertEqual(args.granularity, 20)
        self.assertIsNone(args.rashomon_dir)
        self.assertFalse(args.no_auto_rashomon)
        self.assertEqual(args.unsafe_cost_threshold, 0.5)
        self.assertEqual(args.bc_hidden_dim, 64)
        self.assertEqual(args.bc_n_hidden, 0)
        self.assertEqual(args.rashomon_n_iters, 2000)
        self.assertEqual(args.projection_distance_norm, "l2")
        self.assertEqual(args.eval_every, 0)
        self.assertEqual(args.eval_episodes_periodic, 10)
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)

    def test_parser_accepts_projection_arguments(self) -> None:
        args = build_parser().parse_args(
            [
                "--env-id",
                "CustomBridgeCrossing-v0",
                "--rashomon-dir",
                "/tmp/rashomon_run",
                "--projection-distance-norm",
                "linf",
                "--eval-every",
                "512",
            ]
        )

        self.assertEqual(args.rashomon_dir, Path("/tmp/rashomon_run"))
        self.assertEqual(args.projection_distance_norm, "linf")
        self.assertEqual(args.eval_every, 512)


if __name__ == "__main__":
    unittest.main()
