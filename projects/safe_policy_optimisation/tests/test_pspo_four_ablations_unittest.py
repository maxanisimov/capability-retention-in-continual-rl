"""Focused regression tests for the controlled PSPO ablation suite."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from core.provably_safe_policy_optimisation.adaptive_safe_ppo import AdaptiveSafePPO
from core.provably_safe_policy_optimisation.adaptive_safe_ppo_v2 import AdaptiveSafePPOV2
from provably_safe_policy_optimisation.regions import OrthotopeRegion
from projects.safe_policy_optimisation.scripts import run_pspo_four_ablations as ablations
from projects.safe_policy_optimisation.stages.compute_shield_rashomon_set import (
    build_base_policy,
    fit_base_policy,
)
from projects.safe_policy_optimisation.stages.train_pspo import parse_args
from projects.safe_policy_optimisation.utils.pspo_launcher import (
    base_policy_artifact_matches,
)


class InitializerAblationTests(unittest.TestCase):
    def test_ce_only_stops_at_any_safe_feasibility(self) -> None:
        dataset = {
            "state": torch.tensor([[1.0]]),
            "actions": torch.tensor([[1.0, 1.0, 0.0]]),
        }

        def model() -> torch.nn.Sequential:
            result = build_base_policy(1, 3, hidden_dim=4, n_hidden=0)
            with torch.no_grad():
                result[0].weight.copy_(torch.tensor([[0.1], [0.2], [0.0]]))
                result[0].bias.zero_()
            return result

        ce_only = fit_base_policy(
            model(), dataset, lr=1e-3, max_epochs=0, batch_size=1,
            seed=0, device="cpu", direct_linear_init=False,
            target_margin=2.0, margin_loss_weight=0.0, margin_mode="any",
            safe_action_entropy_weight=0.0,
        )
        canonical = fit_base_policy(
            model(), dataset, lr=1e-3, max_epochs=0, batch_size=1,
            seed=0, device="cpu", direct_linear_init=False,
            target_margin=2.0, margin_loss_weight=1.0, margin_mode="all",
            safe_action_entropy_weight=0.0,
        )
        self.assertTrue(ce_only["reached_target"])
        self.assertEqual(ce_only["stopping_criterion"], "any_safe_feasibility")
        self.assertGreater(ce_only["final_min_any_margin"], 0.0)
        self.assertFalse(canonical["reached_target"])

    def test_any_mode_accepts_what_all_mode_rejects(self) -> None:
        """The real ce_only failures: greedy action safe, but a safe logit below an unsafe one."""
        dataset = {
            "state": torch.tensor([[1.0]]),
            "actions": torch.tensor([[1.0, 1.0, 0.0]]),
        }

        def model() -> torch.nn.Sequential:
            result = build_base_policy(1, 3, hidden_dim=4, n_hidden=0)
            with torch.no_grad():
                # logits [5, -1, 0]: best safe (5) beats unsafe (0) -> any_margin > 0,
                # worst safe (-1) loses to unsafe (0) -> all_margin < 0.
                result[0].weight.copy_(torch.tensor([[5.0], [-1.0], [0.0]]))
                result[0].bias.zero_()
            return result

        def fit(mode: str) -> dict:
            return fit_base_policy(
                model(), dataset, lr=1e-3, max_epochs=0, batch_size=1,
                seed=0, device="cpu", direct_linear_init=False,
                target_margin=2.0, margin_loss_weight=0.0, margin_mode=mode,
                safe_action_entropy_weight=0.0,
            )

        any_mode, all_mode = fit("any"), fit("all")
        self.assertEqual(any_mode["final_accuracy"], 1.0)
        self.assertGreater(any_mode["final_min_any_margin"], 0.0)
        self.assertLess(any_mode["final_min_all_margin"], 0.0)
        self.assertTrue(any_mode["reached_target"])
        self.assertEqual(any_mode["stopping_criterion"], "any_safe_feasibility")
        # Same weights, same accuracy, rejected under the old criterion.
        self.assertFalse(all_mode["reached_target"])
        self.assertEqual(all_mode["stopping_criterion"], "strict_all_safe_feasibility")

    def test_base_policy_cache_separates_margin_loss_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shield = root / "shield.pt"
            shield.write_bytes(b"shield")
            (root / "base_policy.pt").write_bytes(b"policy")
            (root / "safe_behaviour_dataset.pt").write_bytes(b"dataset")
            summary = {
                "base_policy_only": True,
                "shield_sha256": ablations.sha256(shield),
                "architecture": {
                    "hidden_dim": 64, "n_hidden": 2,
                    "state_representation": "one_hot_discrete_observation",
                },
                "dataset": {"dataset_size": 3},
                "base_policy": {
                    "initialisation_objective": "margin",
                    "bc_margin_mode": "all", "target_margin": 2.0,
                    "margin_loss_weight": 0.0,
                    "safe_action_entropy_weight": 0.0,
                    "reached_target": True,
                },
            }
            (root / "summary.json").write_text(json.dumps(summary))
            common = dict(
                shield_path=shield, dataset_size=3, hidden_dim=64, n_hidden=2,
                state_representation="one_hot_discrete_observation",
                margin_mode="all", target_margin=2.0,
                safe_action_entropy_weight=0.0,
            )
            self.assertTrue(
                base_policy_artifact_matches(root, margin_loss_weight=0.0, **common)
            )
            self.assertFalse(
                base_policy_artifact_matches(root, margin_loss_weight=1.0, **common)
            )


class RegionRefreshAndAuditTests(unittest.TestCase):
    def test_fixed_refresh_retains_rollout_enforcement_frequency(self) -> None:
        args = parse_args(
            [
                "--base-policy-path", "base.pt", "--shield-path", "shield.pt",
                "--region-refresh", "fixed", "--freq", "100",
                "--directional", "false", "--env-id", "CustomMiniPacman-v0",
            ]
        )
        self.assertTrue(args.compute_region_once)
        self.assertEqual(args.adaptive_granularity, "train_phase")
        self.assertEqual(args.adaptive_frequency, 100)
        self.assertFalse(args.stop_when_proposal_contained)

    def test_exact_candidate_audit_does_not_mutate_live_actor(self) -> None:
        instance = object.__new__(AdaptiveSafePPO)
        live = torch.nn.Linear(1, 3, bias=False)
        frozen = torch.nn.Linear(1, 3, bias=False)
        instance._live_actor_params = list(live.parameters())
        instance._frozen_actor_seq = frozen
        instance._frozen_actor_params = list(frozen.parameters())
        instance._dataset_states = torch.tensor([[1.0]])
        instance._dataset_actions = torch.tensor([[1.0, 1.0, 0.0]])
        instance._rashomon_multi_label_mode = "all"
        before = [parameter.detach().clone() for parameter in instance._live_actor_params]
        candidate = [torch.tensor([[0.2], [0.1], [0.0]])]
        self.assertTrue(instance._audit_params_greedy_safe(candidate))
        for actual, expected in zip(instance._live_actor_params, before):
            torch.testing.assert_close(actual, expected)

    def test_fixed_region_classifies_safe_excluded_candidate_as_false_negative(self) -> None:
        instance = object.__new__(AdaptiveSafePPOV2)
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        instance._live_actor_params = [parameter]
        instance._last_safe_params = [torch.tensor([0.0])]
        instance._active_regions = [
            OrthotopeRegion(lower=[torch.tensor([-1.0])], upper=[torch.tensor([1.0])])
        ]
        instance.policy = SimpleNamespace(optimizer=SimpleNamespace(_distance_norm="l2"))
        instance._projection_wall_time_s = 0.0
        instance._safety_enforcement_wall_time_s = 0.0
        instance._n_projections = 0
        instance._phase_projections = 0
        instance._last_projection_result = None
        instance._last_phase_projection_result = None
        instance._region_first_false_negatives = 0
        instance._safety_update_events = []
        instance.num_timesteps = 100
        instance._enforce_fixed_region_candidate(
            [parameter.detach().clone()], exact_safe=True, audit_s=0.01
        )
        self.assertEqual(instance._region_first_false_negatives, 1)
        self.assertEqual(instance._safety_update_events[0]["decision"], "projected")
        self.assertTrue(instance._safety_update_events[0]["false_negative"])
        self.assertEqual(instance._safety_update_events[0]["diagnostic_audit_s"], 0.01)
        self.assertEqual(instance._safety_update_events[0]["lid_s"], 0.0)
        self.assertGreaterEqual(
            instance._safety_update_events[0]["safety_enforcement_s"],
            instance._safety_update_events[0]["projection_s"],
        )
        torch.testing.assert_close(parameter, torch.tensor([1.0]))

    def test_fixed_train_phase_projects_repeatedly_without_recomputing_lid(self) -> None:
        instance = object.__new__(AdaptiveSafePPOV2)
        instance._compute_region_once = True
        instance._phase_candidate_params = mock.Mock(return_value=[torch.tensor([0.5])])
        instance._audit_candidate = mock.Mock(return_value=(None, 0.0))
        instance._enforce_fixed_region_candidate = mock.Mock()
        instance._compute_rashomon_around_last_safe = mock.Mock()

        instance._on_train_phase_end()
        instance._on_train_phase_end()

        self.assertEqual(instance._enforce_fixed_region_candidate.call_count, 2)
        instance._compute_rashomon_around_last_safe.assert_not_called()

    def test_non_directional_adaptive_growth_receives_no_proposal_information(self) -> None:
        instance = object.__new__(AdaptiveSafePPOV2)
        parameter = torch.nn.Parameter(torch.tensor([0.5]))
        instance._live_actor_params = [parameter]
        instance._last_safe_params = [torch.tensor([0.0])]
        instance._compute_region_once = False
        instance._directional_rashomon_growth = False
        instance._stop_when_proposal_contained = False
        instance._active_regions = []
        instance._directional_initial_region_pending = False
        instance._rashomon_initial_n_iters = 200
        instance._rashomon_recompute_n_iters = 200
        instance._phase_region_computations = 0
        instance._initial_region_computations = 0
        instance._directional_initial_region_computations = 0
        instance._initial_region_failures = 0
        instance._region_recompute_failures = 0
        instance._phase_projection_failures = 0
        instance._rashomon_wall_time_s = 0.0
        instance._audit_candidate = mock.Mock(return_value=(None, 0.0))
        instance._rashomon_iters_remaining = mock.Mock(return_value=None)
        instance._compute_rashomon_around_last_safe = mock.Mock(return_value=None)
        instance._copy_live_actor_params_from = mock.Mock()
        instance._record_region_first_event = mock.Mock()

        with self.assertWarns(UserWarning):
            instance._on_train_phase_end()

        kwargs = instance._compute_rashomon_around_last_safe.call_args.kwargs
        self.assertIsNone(kwargs["param_l_mask"])
        self.assertIsNone(kwargs["param_u_mask"])
        self.assertIsNone(kwargs["param_objective_weights"])
        self.assertIsNone(kwargs["stop_target_params"])


class OrchestratorTests(unittest.TestCase):
    def _write_control(self, root: Path) -> Path:
        environment = "media_streaming"
        cohort = ablations.CONTROL_COHORTS[environment]
        source = root / cohort / "two_hidden" / environment
        base_dir = source / "initial_base_policy"
        base_dir.mkdir(parents=True)
        base = base_dir / "base_policy.pt"
        base.write_bytes(b"base")
        (base_dir / "summary.json").write_text("{}")
        seed_dir = source / "seed0"
        seed_dir.mkdir()
        adaptive = {
            "granularity": "train_phase", "rollout_interval": 1,
            "rashomon_n_iters": 200, "rashomon_multi_label_mode": "all",
            "rashomon_surrogate": "logsumexp", "rashomon_objective": "weighted_width",
            "region_update_mode": "replace", "directional_rashomon_growth": True,
            "stop_when_proposal_contained": True, "verify_first": False,
        }
        config = {
            "env_id": "env", "env_kwargs": {}, "max_episode_steps": 40,
            "base_policy_path": str(base), "base_policy_architecture": {},
            "total_timesteps": 25_000, "evaluation_policy": "unshielded",
            "training_hyperparameters": {}, "adaptive": adaptive,
        }
        (seed_dir / "config.json").write_text(json.dumps(config))
        (seed_dir / "summary.json").write_text(
            json.dumps({"adaptive_diagnostics": {"rashomon_iters_spent": 1234}})
        )
        (seed_dir / "metrics.json").write_text("{}")
        return source

    def test_budget_extraction_and_variant_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_control(root)
            with mock.patch.object(ablations, "RUNS", root):
                source = ablations.validate_control(
                    "media_streaming", architecture="two_hidden", seeds=[0]
                )
            self.assertEqual(source["seed_budgets"], {"0": 1234})
            fixed = ablations.variant_environment(
                "fixed_lid", "media_streaming", 0, cpu=0,
                architecture="two_hidden", source=source,
                output_root=root / "out", dry_run=True,
                total_timesteps_override=None,
                smoke=False,
            )
            no_gradient = ablations.variant_environment(
                "no_gradient", "media_streaming", 0, cpu=0,
                architecture="two_hidden", source=source,
                output_root=root / "out", dry_run=True,
                total_timesteps_override=None,
                smoke=False,
            )
            no_entropy = ablations.variant_environment(
                "no_entropy", "media_streaming", 0, cpu=0,
                architecture="two_hidden", source=source,
                output_root=root / "out", dry_run=True,
                total_timesteps_override=None, smoke=False,
            )
            ce_only = ablations.variant_environment(
                "ce_only", "media_streaming", 0, cpu=0,
                architecture="two_hidden", source=source,
                output_root=root / "out", dry_run=True,
                total_timesteps_override=None, smoke=False,
            )
            self.assertEqual(fixed.env["REGION_REFRESH"], "fixed")
            self.assertEqual(fixed.env["RASHOMON_N_ITERS"], "1234")
            self.assertEqual(fixed.env["ADAPTIVE_FREQ"], "1")
            self.assertEqual(fixed.env["DIRECTIONAL_RASHOMON_GROWTH"], "0")
            self.assertEqual(no_gradient.env["REGION_REFRESH"], "adaptive")
            self.assertEqual(no_gradient.env["RASHOMON_N_ITERS"], "200")
            self.assertEqual(no_gradient.env["DIRECTIONAL_RASHOMON_GROWTH"], "0")
            self.assertEqual(no_entropy.env["BC_MARGIN_LOSS_WEIGHT"], "1.0")
            self.assertEqual(no_entropy.env["BC_SAFE_ACTION_ENTROPY_WEIGHT"], "0.0")
            self.assertEqual(ce_only.env["BC_MARGIN_LOSS_WEIGHT"], "0.0")
            self.assertEqual(ce_only.env["BC_SAFE_ACTION_ENTROPY_WEIGHT"], "0.0")
            self.assertNotIn("BASE_POLICY_PATH", no_entropy.env)
            self.assertNotIn("BASE_POLICY_PATH", ce_only.env)


if __name__ == "__main__":
    unittest.main()
