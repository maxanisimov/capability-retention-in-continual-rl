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
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    safe_action_margin_loss,
    safe_action_logit_interval_analysis_from_bounds,
    safe_action_margins,
)
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    calibrate_inverse_temperature as calibrate_rashomon_inverse_temperature,
)
from projects.safe_policy_optimisation.utils.masa_env import (
    SafetyBoundArrayWrapper,
)
from projects.safe_policy_optimisation.stages.train_pspo_precomputed import (
    ExecutedActionSafetyCounterWrapper,
    align_rashomon_bounds_to_ppo_actor,
    align_zonotope_region_to_ppo_actor,
    episode_success,
    initialise_ppo_actor_from_base_policy,
    policy_kwargs_from_base_architecture,
    validate_rashomon_shapes,
)
from provably_safe_policy_optimisation import ZonotopeRegion
from projects.safe_policy_optimisation.utils.episode_recording import (
    EpisodeRecorderWrapper,
)
from projects.safe_policy_optimisation.stages.train_ppo_shield import (
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


class RashomonSurrogateCalibrationTests(unittest.TestCase):
    def test_probability_calibration_uses_each_states_safe_action_cardinality(self) -> None:
        dataset = {
            "state": torch.tensor([[0.8, 0.0, 0.0], [0.4, 0.4, 0.0]]),
            "actions": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        }

        inverse_temp, min_valid_mass, corresponding_threshold = (
            calibrate_rashomon_inverse_temperature(
                torch.nn.Identity(),
                dataset,
                inverse_temp_start=1,
                inverse_temp_max=1,
                device="cpu",
            )
        )

        self.assertEqual(inverse_temp, 1)
        self.assertGreaterEqual(min_valid_mass, corresponding_threshold)
        self.assertAlmostEqual(corresponding_threshold, 0.5)

    def test_logsumexp_calibration_uses_selected_surrogate(self) -> None:
        dataset = {
            "state": torch.tensor([[0.5, 0.0], [0.0, 0.5]]),
            "actions": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }

        inverse_temp, min_margin, threshold = calibrate_rashomon_inverse_temperature(
            torch.nn.Identity(),
            dataset,
            inverse_temp_start=1,
            inverse_temp_max=10,
            device="cpu",
            surrogate="logsumexp",
        )

        self.assertGreaterEqual(inverse_temp, 1)
        self.assertGreater(min_margin, 0.0)
        self.assertEqual(threshold, 0.0)

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

    def test_episode_recorder_counts_unsafe_visits_per_step(self) -> None:
        # Regression: unsafe_state_visit_count must accumulate over every step, not
        # only the initial reset state. costs=[0, 1, 0, 2] -> two unsafe steps.
        class _CostSequenceEnv(gym.Env):
            observation_space = gym.spaces.Discrete(1)
            action_space = gym.spaces.Discrete(1)

            def __init__(self, costs: list[float]) -> None:
                super().__init__()
                self._costs = list(costs)
                self._i = 0

            def reset(self, *, seed=None, options=None):
                del seed, options
                self._i = 0
                return 0, {}

            def step(self, action):
                cost = self._costs[self._i]
                self._i += 1
                terminated = self._i >= len(self._costs)
                return 0, 0.0, terminated, False, {"cost": cost}

        env = EpisodeRecorderWrapper(_CostSequenceEnv([0.0, 1.0, 0.0, 2.0]), cost_limit=0.0)
        env.reset()
        done = False
        while not done:
            _obs, _reward, terminated, truncated, _info = env.step(0)
            done = bool(terminated or truncated)

        (episode,) = env.episodes
        self.assertEqual(episode["unsafe_state_visit_count"], 2)
        self.assertFalse(episode["safe_trajectory"])
        self.assertEqual(episode["cost"], 3.0)
        self.assertTrue(episode["violated"])

    def test_generic_safe_rl_env_factory_uses_requested_env(self) -> None:
        env = make_safe_rl_env(
            "CustomMediaStreaming-v0",
            max_episode_steps=5,
            env_kwargs={"fast_rate": 0.0, "slow_rate": 0.0, "out_rate": 0.0},
        )
        try:
            self.assertEqual(env.unwrapped.spec.id, "CustomMediaStreaming-v0")
            self.assertEqual(env.unwrapped._n_states, 20)
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

    def test_bc_margin_mode_all_requires_every_safe_action_to_beat_unsafe(self) -> None:
        logits = torch.tensor([[5.0, 0.0, 4.0]])
        safe_actions = torch.tensor([[1.0, 1.0, 0.0]])

        any_margins, any_contested = safe_action_margins(logits, safe_actions, mode="any")
        all_margins, all_contested = safe_action_margins(logits, safe_actions, mode="all")

        self.assertTrue(bool(any_contested.item()))
        self.assertTrue(bool(all_contested.item()))
        self.assertEqual(float(any_margins.item()), 1.0)
        self.assertEqual(float(all_margins.item()), -4.0)
        self.assertEqual(
            float(safe_action_margin_loss(logits, safe_actions, target_margin=1.0, mode="any").item()),
            0.0,
        )
        self.assertEqual(
            float(safe_action_margin_loss(logits, safe_actions, target_margin=1.0, mode="all").item()),
            5.0,
        )

    def test_safe_action_logit_analysis_separates_surrogate_from_argmax_diversity(self) -> None:
        logits_l = torch.tensor(
            [
                [1.0, 0.0, -2.0],
                [2.0, -1.0, -3.0],
                [0.0, -2.0, -5.0],
            ]
        )
        logits_u = torch.tensor(
            [
                [2.0, 0.5, -1.0],
                [3.0, -0.5, -2.0],
                [1.0, -1.0, -4.0],
            ]
        )
        safe_actions = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        analysis = safe_action_logit_interval_analysis_from_bounds(
            logits_l,
            logits_u,
            safe_actions,
        )

        self.assertEqual(analysis["safe_vs_unsafe_count"], 5)
        self.assertEqual(analysis["total_safe_state_actions"], 5)
        self.assertEqual(analysis["safe_vs_unsafe_micro_pct"], 100.0)
        self.assertEqual(analysis["possible_argmax_count"], 3)
        self.assertEqual(analysis["possible_argmax_micro_pct"], 60.0)
        self.assertEqual(
            analysis["breakdown_by_safe_action_count"]["2"]["possible_argmax_count"],
            2,
        )

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

    def test_base_architecture_maps_to_baseline_mlp_for_two_hidden_layers(self) -> None:
        # The default architecture: PSPO's actor and critic must come out as the
        # same [64, 64] Tanh MLP every baseline uses, so the comparison isolates
        # the safety mechanism rather than network capacity.
        policy_kwargs = policy_kwargs_from_base_architecture(
            {
                "input_dim": 900,
                "n_actions": 5,
                "hidden_dim": 64,
                "n_hidden": 2,
                "activation": "Tanh",
            }
        )

        self.assertEqual(policy_kwargs["net_arch"], [64, 64])

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
            mask = np.ones((env.unwrapped._n_states, env.action_space.n), dtype=int)
            model = ProvablySafePPO(
                "MlpPolicy",
                env,
                shield=mask,
                policy_kwargs=policy_kwargs_from_base_architecture(
                    {
                        "input_dim": env.unwrapped._n_states,
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
                    "input_dim": env.unwrapped._n_states,
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

    def test_align_zonotope_region_reorders_sequential_columns_to_ppo_actor(self) -> None:
        architecture = {
            "input_dim": 2,
            "n_actions": 2,
            "hidden_dim": 64,
            "n_hidden": 0,
            "activation": "Tanh",
        }
        weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        bias = torch.tensor([4.0, 5.0])
        region = ZonotopeRegion(
            center_params=[weight, bias],
            generators=torch.arange(6, dtype=torch.float32).reshape(1, 6),
            coefficient_l=torch.tensor([-1.0]),
            coefficient_u=torch.tensor([1.0]),
            param_shapes=[tuple(weight.shape), tuple(bias.shape)],
        )

        aligned = align_zonotope_region_to_ppo_actor(architecture, region)

        self.assertEqual(aligned.param_shapes, [(2,), (2, 2)])
        self.assertTrue(torch.equal(aligned.center_params[0], bias))
        self.assertTrue(torch.equal(aligned.center_params[1], weight))
        self.assertTrue(
            torch.equal(aligned.generators, torch.tensor([[4.0, 5.0, 0.0, 1.0, 2.0, 3.0]]))
        )

    def test_precomputed_pspo_starts_exactly_from_bc_policy_before_attaching_bounds(self) -> None:
        from provably_safe_policy_optimisation import (
            ProvablySafePPO,
            projection_target_parameter_names,
        )

        architecture = {
            "input_dim": 2,
            "n_actions": 2,
            "hidden_dim": 4,
            "n_hidden": 2,
            "activation": "Tanh",
        }
        torch.manual_seed(123)
        base_policy = build_shield_rashomon_policy(
            input_dim=architecture["input_dim"],
            n_actions=architecture["n_actions"],
            hidden_dim=architecture["hidden_dim"],
            n_hidden=architecture["n_hidden"],
        )
        base_state_dict = {
            name: parameter.detach().clone()
            for name, parameter in base_policy.state_dict().items()
        }

        env = TwoStateEnv()
        try:
            model = ProvablySafePPO(
                "MlpPolicy",
                env,
                shield=np.ones((2, 2), dtype=int),
                policy_kwargs=policy_kwargs_from_base_architecture(architecture),
                n_steps=8,
                batch_size=4,
                n_epochs=1,
                verbose=0,
            )
            diagnostics = initialise_ppo_actor_from_base_policy(
                model,
                architecture,
                base_state_dict,
            )

            observations = torch.eye(2)
            with torch.no_grad():
                expected_logits = base_policy(observations)
                actor_latent = model.policy.mlp_extractor.policy_net(observations)
                actual_logits = model.policy.action_net(actor_latent)
            self.assertTrue(torch.equal(actual_logits, expected_logits))
            self.assertEqual(diagnostics["max_abs_parameter_error"], 0.0)

            target_names = projection_target_parameter_names(model)
            named_params = dict(model.policy.named_parameters())
            actor_before_bounds = {
                name: named_params[name].detach().clone() for name in target_names
            }
            raw_lower = [
                parameter.detach().clone() - 0.1 for parameter in base_policy.parameters()
            ]
            raw_upper = [
                parameter.detach().clone() + 0.1 for parameter in base_policy.parameters()
            ]
            lower, upper = align_rashomon_bounds_to_ppo_actor(
                architecture,
                raw_lower,
                raw_upper,
            )
            validate_rashomon_shapes(model, lower, upper)
            model.set_projection_bounds(lower, upper, project_on_set=False)

            self.assertTrue(model.is_within_bounds())
            for name in target_names:
                self.assertTrue(torch.equal(named_params[name], actor_before_bounds[name]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
