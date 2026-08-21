"""Tests for AdaptiveSafePPOV2 projection-triggered region updates."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th

from provably_safe_policy_optimisation import AdaptiveSafePPOV2
from provably_safe_policy_optimisation import adaptive_safe_ppo as asp
from provably_safe_policy_optimisation.adaptive_safe_ppo_v2 import (
    _directional_masks_from_update_deltas,
)


def _only_action_safe(action: int, n_states: int = 16, n_actions: int = 4) -> np.ndarray:
    mask = np.zeros((n_states, n_actions), dtype=int)
    mask[:, action] = 1
    return mask


def _safe_base_state_dict(action: int = 1) -> dict[str, th.Tensor]:
    bias = th.zeros(4)
    bias[action] = 5.0
    return {"action_net.weight": th.zeros(4, 16), "action_net.bias": bias}


def _stub_engine_sequence(
    record: list,
    certificates: list[float],
    *,
    eps: float = 0.05,
    cover_stop_target: bool = False,
    early_iterations: int = 3,
):
    state = {"calls": 0}

    def stub(model, dataset, **kwargs):  # type: ignore[no-untyped-def]
        del dataset
        params = [p.detach().clone() for p in model.parameters()]
        index = min(state["calls"], len(certificates) - 1)
        state["calls"] += 1
        record.append({"params": params, "kwargs": kwargs})
        lower = [p - eps for p in params]
        upper = [p + eps for p in params]
        stop_target = kwargs.get("stop_target_params")
        target_covered = bool(cover_stop_target and stop_target is not None)
        if target_covered:
            lower = [th.minimum(bound, target) for bound, target in zip(lower, stop_target)]
            upper = [th.maximum(bound, target) for bound, target in zip(upper, stop_target)]
        bounded = SimpleNamespace(param_l=lower, param_u=upper)
        return SimpleNamespace(
            bounded_models=[bounded],
            certificates=[[SimpleNamespace(min_hard_acc=certificates[index])]],
            iterations_run=(early_iterations if target_covered else kwargs["n_iters"]),
            target_contained_and_certified=target_covered,
        )

    return stub


class AdaptiveSafePPOV2Tests(unittest.TestCase):
    def _patch_engine(
        self,
        record: list,
        certificates: list[float] | None = None,
        **stub_kwargs,
    ) -> None:
        original = asp._run_rashomon_engine
        asp._run_rashomon_engine = _stub_engine_sequence(
            record, certificates or [1.0], **stub_kwargs
        )
        self.addCleanup(setattr, asp, "_run_rashomon_engine", original)

    def _make(self, **extra) -> AdaptiveSafePPOV2:
        env = gym.make("FrozenLake-v1")
        extra.setdefault("base_policy_state_dict", _safe_base_state_dict())
        extra.setdefault("n_steps", 32)
        extra.setdefault("batch_size", 16)
        extra.setdefault("n_epochs", 1)
        # Preserve the legacy non-directional setup unless a test opts into
        # the unified method's new directional default explicitly.
        extra.setdefault("directional_rashomon_growth", False)
        extra.setdefault("stop_when_proposal_contained", False)
        model = AdaptiveSafePPOV2(
            "MlpPolicy",
            env,
            shield=_only_action_safe(1),
            seed=0,
            shield_seed=0,
            device="cpu",
            verbose=0,
            policy_kwargs={"net_arch": []},
            **extra,
        )
        self.addCleanup(model.get_env().close)
        return model

    def _force_projection(self, model: AdaptiveSafePPOV2) -> None:
        with th.no_grad():
            model._live_actor_params[0].data.add_(1.0)
        proposed = [param.detach().clone() for param in model._live_actor_params]
        result = model.project_now()
        self.assertGreater(result.n_projected, 0)
        model.policy.optimizer._last_proposed_params = proposed
        model.policy.optimizer._last_proposed_update_deltas = [
            target - snapshot
            for target, snapshot in zip(proposed, model._last_safe_params)
        ]
        model._on_gradient_step()

    def test_initial_region_is_computed_once(self) -> None:
        record: list = []
        self._patch_engine(record)

        model = self._make()

        diag = model.adaptive_diagnostics()
        self.assertEqual(len(record), 1)
        self.assertEqual(diag["initial_region_computations"], 1)
        self.assertEqual(diag["current_region_count"], 1)
        self.assertEqual(diag["rashomon_computations"], 1)

    def test_train_phase_granularity_skips_step_hook_and_initial_region(self) -> None:
        record: list = []
        self._patch_engine(record)

        model = self._make(adaptive_granularity="train_phase")

        diag = model.adaptive_diagnostics()
        self.assertEqual(record, [])
        self.assertEqual(getattr(model.policy.optimizer, "post_step_hook", None), None)
        self.assertEqual(diag["granularity"], "train_phase")
        self.assertEqual(diag["initial_region_computations"], 0)
        self.assertEqual(diag["current_region_count"], 0)

    def test_train_phase_interval_flushes_pending_aggregate_update(self) -> None:
        record: list = []
        self._patch_engine(record)
        model = self._make(
            adaptive_granularity="train_phase",
            adaptive_frequency=3,
            directional_rashomon_growth=True,
            stop_when_proposal_contained=True,
        )
        model.learn(total_timesteps=32)
        self.assertEqual(record, [])
        self.assertTrue(model.adaptive_diagnostics()["pending_adaptive_update"])
        model.finalize_adaptive_update()
        self.assertEqual(len(record), 1)
        self.assertEqual(model.adaptive_diagnostics()["final_flushes"], 1)

    def test_compute_region_once_never_recomputes_after_optimizer_step(self) -> None:
        record: list = []
        self._patch_engine(record)
        model = self._make(compute_region_once=True)
        actor_param = model._live_actor_params[0]
        actor_param.grad = th.zeros_like(actor_param)
        model.policy.optimizer.step()
        self.assertEqual(len(record), 1)
        self.assertTrue(model.adaptive_diagnostics()["compute_region_once"])

    def test_train_phase_directional_growth_uses_full_phase_delta_and_projects_once(self) -> None:
        record: list = []
        self._patch_engine(record, [1.0], eps=0.05)
        model = self._make(
            adaptive_granularity="train_phase",
            directional_rashomon_growth=True,
        )
        safe_weight = model._last_safe_params[0].detach().clone()

        with th.no_grad():
            model._live_actor_params[0].data.add_(0.20)
        model._on_train_phase_end()

        self.assertEqual(len(record), 1)
        lower = record[0]["kwargs"]["param_l_mask"]
        upper = record[0]["kwargs"]["param_u_mask"]
        objective_weights = record[0]["kwargs"]["param_objective_weights"]
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(objective_weights)
        self.assertTrue(bool(lower[0].all()))
        self.assertFalse(bool(upper[0].any()))
        self.assertTrue(th.allclose(objective_weights[0], th.full_like(safe_weight, 0.20)))
        for weight in objective_weights[1:]:
            self.assertTrue(th.equal(weight, th.zeros_like(weight)))
        expected = safe_weight + 0.05
        self.assertTrue(th.allclose(model._live_actor_params[0].detach(), expected))
        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["phase_region_computations"], 1)
        self.assertEqual(diag["phase_projections"], 1)
        self.assertEqual(diag["projection_triggers"], 0)
        self.assertEqual(diag["initial_region_computations"], 1)
        self.assertEqual(diag["directional_initial_region_computations"], 1)
        self.assertFalse(diag["directional_initial_region_pending"])

    def test_directional_mask_sign_convention_and_zero_freezing(self) -> None:
        lower, upper, counts = _directional_masks_from_update_deltas(
            [th.tensor([1.0, -2.0, 0.0])]
        )

        self.assertTrue(th.equal(lower[0], th.tensor([True, False, True])))
        self.assertTrue(th.equal(upper[0], th.tensor([False, True, True])))
        self.assertEqual(counts, {"positive": 1, "negative": 1, "zero": 1})

    def test_directional_growth_forwards_full_proposed_update_masks(self) -> None:
        record: list = []
        self._patch_engine(record, [1.0])
        model = self._make(directional_rashomon_growth=True, learning_rate=1.0)

        self.assertEqual(record, [])
        before = model.adaptive_diagnostics()
        self.assertTrue(before["directional_initial_region_pending"])
        self.assertEqual(before["initial_region_computations"], 0)
        point_region = model._active_regions[0]
        self.assertTrue(
            all(
                th.equal(lower, param.detach()) and th.equal(upper, param.detach())
                for lower, param, upper in zip(
                    point_region.lower, model._live_actor_params, point_region.upper
                )
            )
        )

        actor_param = model._live_actor_params[0]
        actor_param.grad = th.zeros_like(actor_param)
        flat_grad = actor_param.grad.reshape(-1)
        flat_grad[:3] = th.tensor([-1.0, 1.0, 0.0])
        model.policy.optimizer.step()

        self.assertEqual(len(record), 1)
        lower = record[0]["kwargs"]["param_l_mask"]
        upper = record[0]["kwargs"]["param_u_mask"]
        objective_weights = record[0]["kwargs"]["param_objective_weights"]
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(objective_weights)
        self.assertTrue(th.equal(lower[0].reshape(-1)[:3], th.tensor([True, False, True])))
        self.assertTrue(th.equal(upper[0].reshape(-1)[:3], th.tensor([False, True, True])))
        self.assertTrue(
            th.allclose(
                objective_weights[0].reshape(-1)[:3],
                th.tensor([1.0, 1.0, 0.0]),
            )
        )
        diagnostics = model.adaptive_diagnostics()
        self.assertFalse(diagnostics["directional_initial_region_pending"])
        self.assertEqual(diagnostics["initial_region_computations"], 1)
        self.assertEqual(diagnostics["directional_initial_region_computations"], 1)
        self.assertEqual(diagnostics["directional_region_recomputations"], 0)
        self.assertEqual(diagnostics["region_recomputations"], 0)
        self.assertEqual(diagnostics["directional_growth_failures"], 0)
        self.assertGreater(diagnostics["last_direction_counts"]["zero"], 0)

    def test_directional_growth_without_optimizer_delta_keeps_previous_region(self) -> None:
        record: list = []
        self._patch_engine(record)
        model = self._make(directional_rashomon_growth=True)

        with th.no_grad():
            model._live_actor_params[0].data.add_(1.0)
        result = model.project_now()
        self.assertGreater(result.n_projected, 0)
        with self.assertWarnsRegex(UserWarning, "could not retain the proposed"):
            model._on_gradient_step()

        diagnostics = model.adaptive_diagnostics()
        self.assertEqual(len(record), 0)
        self.assertEqual(diagnostics["current_region_count"], 1)
        self.assertEqual(diagnostics["initial_region_failures"], 1)
        self.assertTrue(diagnostics["directional_initial_region_pending"])
        self.assertEqual(diagnostics["directional_growth_failures"], 1)

    def test_stops_growth_and_accepts_exact_proposal_after_certified_containment(self) -> None:
        record: list = []
        self._patch_engine(
            record,
            [1.0, 1.0],
            cover_stop_target=True,
            early_iterations=3,
        )
        model = self._make(
            directional_rashomon_growth=True,
            stop_when_proposal_contained=True,
            rashomon_n_iters=10,
            learning_rate=1.0,
        )

        actor_param = model._live_actor_params[0]
        actor_param.grad = th.zeros_like(actor_param)
        actor_param.grad.reshape(-1)[0] = -1.0
        model.policy.optimizer.step()

        proposal = record[0]["kwargs"]["stop_target_params"]
        self.assertIsNotNone(proposal)
        for actual, target in zip(model._live_actor_params, proposal):
            self.assertTrue(th.equal(actual.detach(), target))
        diagnostics = model.adaptive_diagnostics()
        self.assertEqual(diagnostics["proposal_containment_early_stops"], 1)
        self.assertEqual(diagnostics["proposed_updates_accepted_after_growth"], 1)
        self.assertEqual(diagnostics["rashomon_iterations_saved_by_containment"], 7)
        self.assertEqual(diagnostics["rashomon_iters_spent"], 3)
        self.assertEqual(diagnostics["initial_region_iters_spent"], 3)
        self.assertEqual(diagnostics["region_recompute_iters_spent"], 0)
        self.assertEqual(diagnostics["directional_initial_region_computations"], 1)

    def test_directional_growth_rejects_zonotope_regions(self) -> None:
        env = gym.make("FrozenLake-v1")
        self.addCleanup(env.close)
        with self.assertRaisesRegex(ValueError, "requires safe_region_shape='orthotope'"):
            AdaptiveSafePPOV2(
                "MlpPolicy",
                env,
                shield=_only_action_safe(1),
                base_policy_state_dict=_safe_base_state_dict(),
                directional_rashomon_growth=True,
                safe_region_shape="zonotope",
                policy_kwargs={"net_arch": []},
                n_steps=32,
                batch_size=16,
                n_epochs=1,
            )

    def test_inside_step_does_not_recompute_region(self) -> None:
        record: list = []
        self._patch_engine(record)
        model = self._make()

        model.project_now()
        model._on_gradient_step()

        diag = model.adaptive_diagnostics()
        self.assertEqual(len(record), 1)
        self.assertEqual(diag["projection_triggers"], 0)
        self.assertEqual(diag["region_recomputations"], 0)

    def test_union_mode_appends_region_after_active_projection(self) -> None:
        record: list = []
        self._patch_engine(record, [1.0, 1.0])
        model = self._make(region_update_mode="union")

        self._force_projection(model)

        diag = model.adaptive_diagnostics()
        self.assertEqual(len(record), 2)
        self.assertEqual(diag["projection_triggers"], 1)
        self.assertEqual(diag["region_recomputations"], 1)
        self.assertEqual(diag["current_region_count"], 2)

    def test_replace_mode_keeps_one_active_region(self) -> None:
        record: list = []
        self._patch_engine(record, [1.0, 1.0])
        model = self._make(region_update_mode="replace")

        self._force_projection(model)

        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["region_recomputations"], 1)
        self.assertEqual(diag["current_region_count"], 1)

    def test_failed_recompute_keeps_existing_region(self) -> None:
        record: list = []
        self._patch_engine(record, [1.0, 0.5])
        model = self._make(region_update_mode="union")

        self._force_projection(model)

        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["region_recomputations"], 0)
        self.assertEqual(diag["region_recompute_failures"], 1)
        self.assertEqual(diag["current_region_count"], 1)

    def test_total_budget_derives_per_computation_iters(self) -> None:
        record: list = []
        self._patch_engine(record)

        model = self._make(
            rashomon_budget_mode="total",
            rashomon_total_iters=55,
            rashomon_max_region_computations=11,
        )

        self.assertEqual(record[0]["kwargs"]["n_iters"], 55)
        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["rashomon_n_iters"], 55)
        self.assertEqual(diag["rashomon_iters_spent"], 55)
        self.assertEqual(diag["rashomon_iters_remaining"], 0)
        self.assertEqual(diag["initial_region_iters_spent"], 55)
        self.assertEqual(diag["region_recompute_iters_spent"], 0)

    def test_total_budget_uses_separate_initial_and_recompute_iters(self) -> None:
        record: list = []
        self._patch_engine(record, [1.0, 1.0, 1.0])

        model = self._make(
            rashomon_budget_mode="total",
            rashomon_total_iters=30,
            rashomon_initial_n_iters=20,
            rashomon_recompute_n_iters=7,
            rashomon_max_region_computations=11,
        )
        self._force_projection(model)
        self._force_projection(model)

        self.assertEqual([call["kwargs"]["n_iters"] for call in record], [20, 7, 3])
        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["projection_triggers"], 2)
        self.assertEqual(diag["region_recomputations"], 2)
        self.assertEqual(diag["rashomon_iters_spent"], 30)
        self.assertEqual(diag["rashomon_iters_remaining"], 0)
        self.assertEqual(diag["initial_region_iters_spent"], 20)
        self.assertEqual(diag["region_recompute_iters_spent"], 10)

    def test_total_budget_exhaustion_skips_region_recompute(self) -> None:
        record: list = []
        self._patch_engine(record, [1.0, 1.0])

        model = self._make(
            rashomon_budget_mode="total",
            rashomon_total_iters=5,
            rashomon_max_region_computations=11,
        )
        self._force_projection(model)

        diag = model.adaptive_diagnostics()
        self.assertEqual(len(record), 1)
        self.assertEqual(diag["projection_triggers"], 1)
        self.assertEqual(diag["region_recomputations"], 0)
        self.assertEqual(diag["region_recompute_failures"], 1)
        self.assertEqual(diag["region_recompute_budget_exhaustions"], 1)


if __name__ == "__main__":
    unittest.main()
