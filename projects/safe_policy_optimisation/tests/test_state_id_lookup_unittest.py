"""Integration tests for the sparse PSPO state representation."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
import torch
from abstract_gradient_training.bounded_models import StateIdLookupLinear
from provably_safe_policy_optimisation import AdaptiveSafePPO
from provably_safe_policy_optimisation.adaptive_safe_ppo import state_id_lookup_actor

from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    build_base_policy,
    initialise_linear_policy_from_masks,
    make_safe_behaviour_payload,
)


class StateIdLookupIntegrationTests(unittest.TestCase):
    def test_payload_stores_ids_and_preserves_logical_input_dimension(self) -> None:
        mask = np.asarray([[1, 0], [0, 0], [0, 1], [1, 1]], dtype=np.float32)
        payload, metadata = make_safe_behaviour_payload(
            mask, state_representation="state_id_lookup"
        )
        self.assertEqual(payload["state"].dtype, torch.long)
        self.assertEqual(payload["state"].tolist(), [[0], [2], [3]])
        self.assertEqual(tuple(payload["actions"].shape), (3, 2))
        self.assertEqual(metadata["input_dim"], 4)
        self.assertEqual(metadata["feature_dim"], 4)
        self.assertEqual(metadata["stored_state_shape"], [3, 1])
        self.assertEqual(metadata["stored_state_bytes"], 3 * 8)
        self.assertEqual(
            metadata["state_representation"],
            "state_id_lookup_discrete_observation",
        )

    def test_lookup_rejects_feature_decoder(self) -> None:
        mask = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            make_safe_behaviour_payload(
                mask,
                lambda state: np.asarray([state], dtype=np.float32),
                state_representation="state_id_lookup",
            )

    def test_tabular_initialisation_matches_dense_one_hot(self) -> None:
        mask = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        dense_data, _ = make_safe_behaviour_payload(mask)
        lookup_data, _ = make_safe_behaviour_payload(
            mask, state_representation="state_id_lookup"
        )
        dense = build_base_policy(3, 2, hidden_dim=4, n_hidden=0)
        lookup = build_base_policy(
            3, 2, hidden_dim=4, n_hidden=0, state_id_lookup=True
        )
        self.assertTrue(initialise_linear_policy_from_masks(dense, dense_data, margin=2))
        self.assertTrue(initialise_linear_policy_from_masks(lookup, lookup_data, margin=2))
        lookup.load_state_dict(dense.state_dict())
        torch.testing.assert_close(
            lookup(lookup_data["state"]),
            dense(dense_data["state"]),
            rtol=0,
            atol=0,
        )

    def test_live_actor_lookup_view_shares_parameters(self) -> None:
        dense = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(5, 3),
            torch.nn.Tanh(),
            torch.nn.Linear(3, 2),
        )
        lookup = state_id_lookup_actor(dense, share_parameters=True)
        self.assertIsInstance(lookup[1], StateIdLookupLinear)
        self.assertIs(lookup[1].weight, dense[1].weight)
        ids = torch.tensor([[0], [4], [2]])
        one_hot = torch.nn.functional.one_hot(ids[:, 0], 5).float()
        torch.testing.assert_close(lookup(ids), dense(one_hot), rtol=0, atol=0)

    def test_adaptive_pspo_keeps_exhaustive_states_as_ids(self) -> None:
        mask = np.zeros((16, 4), dtype=np.float32)
        mask[:, 1] = 1
        bias = torch.zeros(4)
        bias[1] = 5
        env = gym.make("FrozenLake-v1")
        model = AdaptiveSafePPO(
            "MlpPolicy",
            env,
            shield=mask,
            base_policy_state_dict={
                "action_net.weight": torch.zeros(4, 16),
                "action_net.bias": bias,
            },
            discrete_state_representation="state_id_lookup",
            policy_kwargs={"net_arch": []},
            directional_rashomon_growth=False,
            stop_when_proposal_contained=False,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            seed=0,
            device="cpu",
            verbose=0,
        )
        self.addCleanup(model.get_env().close)
        self.assertEqual(model._dataset_states.dtype, torch.long)
        self.assertEqual(tuple(model._dataset_states.shape), (16, 1))
        self.assertIsNone(model._verify_obs)
        self.assertIsInstance(model._live_actor_seq[-1], StateIdLookupLinear)
        self.assertEqual(model._greedy_safe_rate_now(), 1.0)

if __name__ == "__main__":
    unittest.main()
