"""Tests for shielded / Rashomon / MASA-shielded stage helpers."""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    allowed_action_accuracy as shield_rashomon_accuracy,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    build_base_policy as build_shield_rashomon_policy,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    fit_base_policy as fit_shield_rashomon_policy,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    load_shield_mask as load_rashomon_shield_mask,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    make_safe_behaviour_payload,
)
from projects.safe_policy_optimisation.stages.train_masa_shielded_policy import (
    SafetyBoundArrayWrapper,
)
from projects.safe_policy_optimisation.stages.train_rashomon_shielded_policy import (
    ExecutedActionSafetyCounterWrapper,
    align_rashomon_bounds_to_ppo_actor,
    episode_success,
    policy_kwargs_from_base_architecture,
    validate_rashomon_shapes,
)
from projects.safe_policy_optimisation.stages.train_discrete_shielded_policy import (
    load_shield_mask,
    validate_shield_for_env,
)
from projects.safe_policy_optimisation.utils.safe_rl import (
    EpisodeMetrics,
    aggregate_training_violations,
    aggregate_violations,
    build_safe_rl_baseline,
    evaluate_policy,
    make_minipacman_cost_fn,
    make_minipacman_env,
    make_safe_rl_env,
    minipacman_state_cost,
    save_gif,
    training_episode_rows,
)
from projects.safe_policy_optimisation.tests.helpers import (
    TwoStateEnv,
)

class MasaShieldedWrapperTests(unittest.TestCase):
    def test_safety_bound_wrapper_matches_declared_box_shape(self) -> None:
        wrapped = SafetyBoundArrayWrapper.__new__(SafetyBoundArrayWrapper)

        obs = wrapped.observation({"orig_obs": 3, "safety_bound": 0.0})

        self.assertEqual(obs["orig_obs"], 3)
        self.assertEqual(obs["safety_bound"].shape, (1,))
        self.assertEqual(obs["safety_bound"].dtype, np.float32)

class GenericShieldedPolicyTests(unittest.TestCase):
    def test_load_shield_mask_from_binary_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shield_q.pt"
            torch.save({"shield": torch.tensor([[1, 0], [0, 1]])}, path)

            mask = load_shield_mask(path)

            self.assertTrue(np.array_equal(mask, np.array([[1, 0], [0, 1]])))

    def test_load_shield_mask_from_action_risk_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shield_q.pt"
            torch.save({"action_risk": torch.tensor([[0.0, 0.5], [1.0, 0.0]])}, path)

            mask = load_shield_mask(path, source="action_risk", risk_threshold=0.0)

            self.assertTrue(np.array_equal(mask, np.array([[1, 0], [0, 1]])))

    def test_validate_shield_shape_mismatch_raises(self) -> None:
        env = make_minipacman_env(max_episode_steps=5)
        try:
            with self.assertRaises(ValueError):
                validate_shield_for_env(np.ones((2, 2), dtype=int), env)
        finally:
            env.close()

    def test_generic_safe_rl_env_factory_uses_requested_env(self) -> None:
        env = make_safe_rl_env(
            "CustomMediaStreaming-v0",
            max_episode_steps=5,
            env_kwargs={"fast_rate": 0.0, "slow_rate": 0.0, "out_rate": 0.0},
        )
        try:
            self.assertEqual(env.unwrapped.spec.id, "CustomMediaStreaming-v0")
            self.assertEqual(env.observation_space.n, 20)
        finally:
            env.close()

class ShieldRashomonDatasetTests(unittest.TestCase):
    def test_safe_behaviour_payload_uses_one_hot_state_features(self) -> None:
        mask = np.array(
            [
                [1, 0, 1],
                [0, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )

        payload, metadata = make_safe_behaviour_payload(mask)

        self.assertTrue(
            torch.equal(
                payload["state"],
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                payload["actions"],
                torch.tensor(
                    [
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ]
                ),
            )
        )
        self.assertEqual(metadata["excluded_no_safe_action_states"], 1)

    def test_load_rashomon_shield_mask_falls_back_to_action_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shield_q.pt"
            torch.save({"action_risk": torch.tensor([[0.0, 0.2], [0.5, 0.0]]), "risk_threshold": 0.1}, path)

            mask = load_rashomon_shield_mask(path)

            self.assertTrue(np.array_equal(mask, np.array([[1, 0], [0, 1]], dtype=np.float32)))

    def test_linear_base_policy_reaches_perfect_allowed_action_accuracy(self) -> None:
        mask = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )
        payload, _metadata = make_safe_behaviour_payload(mask)
        model = build_shield_rashomon_policy(
            input_dim=payload["state"].shape[1],
            n_actions=payload["actions"].shape[1],
            hidden_dim=4,
            n_hidden=0,
        )

        metrics = fit_shield_rashomon_policy(
            model,
            payload,
            lr=1e-3,
            max_epochs=1,
            batch_size=2,
            seed=0,
            device="cpu",
        )

        self.assertTrue(metrics["reached_target"])
        self.assertEqual(metrics["epochs_run"], 0)
        self.assertEqual(shield_rashomon_accuracy(model, payload, device="cpu"), 1.0)

class RashomonShieldedPPOTests(unittest.TestCase):
    def test_executed_action_counter_counts_action_safety_at_pre_step_state(self) -> None:
        mask = np.array([[1, 0], [0, 1]], dtype=int)
        env = ExecutedActionSafetyCounterWrapper(TwoStateEnv(), mask)

        obs, _info = env.reset()
        self.assertEqual(obs, 0)
        env.step(1)
        env.step(1)

        diagnostics = env.diagnostics()
        self.assertEqual(diagnostics["executed_action_checks"], 2)
        self.assertEqual(diagnostics["executed_unsafe_action_count"], 1)
        self.assertEqual(diagnostics["executed_unsafe_action_percentage"], 50.0)
        self.assertTrue(env.records[0]["unsafe_executed_action"])
        self.assertFalse(env.records[1]["unsafe_executed_action"])

    def test_episode_success_prefers_info_flags_then_reward(self) -> None:
        self.assertTrue(episode_success(0.0, [{"is_success": True}], reward_threshold=0.0))
        self.assertFalse(episode_success(1.0, [{"success": False}], reward_threshold=0.0))
        self.assertTrue(episode_success(1.0, [{}], reward_threshold=0.0))
        self.assertFalse(episode_success(0.0, [{}], reward_threshold=0.0))

    def test_base_architecture_maps_to_empty_ppo_net_for_linear_policy(self) -> None:
        policy_kwargs = policy_kwargs_from_base_architecture(
            {
                "input_dim": 9248,
                "n_actions": 5,
                "hidden_dim": 64,
                "n_hidden": 0,
                "activation": "Tanh",
            }
        )

        self.assertEqual(policy_kwargs["net_arch"], [])

    def test_validate_rashomon_shapes_accepts_linear_minipacman_actor_bounds(self) -> None:
        from provably_safe_policy_optimisation import ProvablySafePPO

        env = make_minipacman_env(max_episode_steps=5)
        try:
            mask = np.ones((env.observation_space.n, env.action_space.n), dtype=int)
            model = ProvablySafePPO(
                "MlpPolicy",
                env,
                shield=mask,
                policy_kwargs=policy_kwargs_from_base_architecture(
                    {
                        "input_dim": env.observation_space.n,
                        "n_actions": env.action_space.n,
                        "hidden_dim": 64,
                        "n_hidden": 0,
                        "activation": "Tanh",
                    }
                ),
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                verbose=0,
            )
            lower = [torch.zeros_like(param.detach()) for param in model.policy.action_net.parameters()]
            upper = [torch.ones_like(param.detach()) for param in model.policy.action_net.parameters()]
            lower, upper = align_rashomon_bounds_to_ppo_actor(
                {
                    "input_dim": env.observation_space.n,
                    "n_actions": env.action_space.n,
                    "hidden_dim": 64,
                    "n_hidden": 0,
                    "activation": "Tanh",
                },
                lower,
                upper,
            )

            rows = validate_rashomon_shapes(model, lower, upper)

            self.assertEqual([row["parameter"] for row in rows], ["action_net.bias", "action_net.weight"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
