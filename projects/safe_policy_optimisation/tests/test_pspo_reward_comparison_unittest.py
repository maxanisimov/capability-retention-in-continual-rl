"""Tests for reproducible PSPO reward and safety comparisons."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from projects.safe_policy_optimisation.scripts.compare_pspo_initial_final_rewards import (
    BasePolicyPredictor,
    build_parser,
    exhaustive_shield_alignment,
)


class _FixedActionPredictor:
    def __init__(self, actions: list[int]) -> None:
        self.actions = np.asarray(actions, dtype=np.int64)

    def predict(self, observations, deterministic=True):
        del deterministic
        states = np.asarray(observations).argmax(axis=1)
        return self.actions[states], None


class _DiscreteFixedActionPredictor(_FixedActionPredictor):
    observation_space = type("DiscreteSpace", (), {"n": 4})()

    def predict(self, observations, deterministic=True):
        del deterministic
        states = np.asarray(observations)
        self.last_observation_shape = states.shape
        return self.actions[states], None


class PspoAdaptiveRewardComparisonTests(unittest.TestCase):
    def test_environment_and_seed_filters_parse(self) -> None:
        args = build_parser().parse_args(
            ["--environments", "bridge_crossing", "mini_pacman", "--seeds", "0", "4"]
        )

        self.assertEqual(args.environments, ["bridge_crossing", "mini_pacman"])
        self.assertEqual(args.seeds, [0, 4])

    def test_base_policy_predictor_supports_batched_one_hot_states(self) -> None:
        policy = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            policy.weight.copy_(torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]]))
        predictor = BasePolicyPredictor(policy, input_dim=3)

        actions, _ = predictor.predict(np.eye(3, dtype=np.float32))

        np.testing.assert_array_equal(actions, np.asarray([0, 1, 0]))

    def test_exhaustive_alignment_ignores_states_without_safe_actions(self) -> None:
        mask = np.asarray(
            [[1, 0], [0, 0], [0, 1], [1, 0]],
            dtype=np.int64,
        )
        predictor = _FixedActionPredictor([0, 1, 0, 0])

        result = exhaustive_shield_alignment(
            predictor,
            mask,
            input_dim=4,
            batch_size=2,
        )

        self.assertEqual(result["checked_states"], 3)
        self.assertEqual(result["unsafe_states"], 1)
        self.assertAlmostEqual(result["alignment_rate"], 2 / 3)

    def test_exhaustive_alignment_passes_state_ids_to_discrete_models(self) -> None:
        predictor = _DiscreteFixedActionPredictor([0, 1, 1, 0])
        mask = np.asarray([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.int64)

        result = exhaustive_shield_alignment(predictor, mask, input_dim=4)

        self.assertEqual(predictor.last_observation_shape, (4,))
        self.assertEqual(result["alignment_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
