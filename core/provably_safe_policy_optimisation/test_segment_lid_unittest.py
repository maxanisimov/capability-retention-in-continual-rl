"""Tests for PSPO's segment safe region (safe_region_shape='segment')."""

from __future__ import annotations

import unittest
import warnings

import gymnasium as gym
import numpy as np
import torch as th

from provably_safe_policy_optimisation import AdaptiveSafePPO, AdaptiveSafePPOV2
from provably_safe_policy_optimisation.regions import (
    ZonotopeRegion,
    flatten_tensors,
)


def _only_action_safe(action: int, n_states: int = 16, n_actions: int = 4) -> np.ndarray:
    mask = np.zeros((n_states, n_actions), dtype=int)
    mask[:, action] = 1
    return mask


def _safe_base_state_dict(action: int = 1) -> dict[str, th.Tensor]:
    bias = th.zeros(4)
    bias[action] = 5.0
    return {"action_net.weight": th.zeros(4, 16), "action_net.bias": bias}


class SegmentLIDTests(unittest.TestCase):
    def _make(self, cls=AdaptiveSafePPO, **extra):
        extra.setdefault("base_policy_state_dict", _safe_base_state_dict())
        extra.setdefault("n_steps", 32)
        extra.setdefault("batch_size", 16)
        extra.setdefault("n_epochs", 1)
        extra.setdefault("learning_rate", 1e-3)
        extra.setdefault("safe_region_shape", "segment")
        extra.setdefault("rashomon_multi_label_mode", "any")
        extra.setdefault("seed", 0)
        return cls(
            "MlpPolicy",
            gym.make("FrozenLake-v1"),
            shield=_only_action_safe(1),
            policy_kwargs={"net_arch": []},
            **extra,
        )

    # ------------------------------------------------------------- validation

    def test_segment_is_an_accepted_region_shape(self) -> None:
        model = self._make()
        self.assertEqual(model.adaptive_diagnostics()["safe_region_shape"], "segment")

    def test_unknown_region_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._make(safe_region_shape="polytope")

    def test_invalid_segment_settings_are_rejected(self) -> None:
        for kwargs in (
            {"segment_tolerance": 0.0},
            {"segment_tolerance": 1.0},
            {"segment_splits": 0},
            {"segment_splits": 4, "segment_max_splits": 2},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    self._make(**kwargs)

    def test_v2_rejects_settings_that_assume_a_volume(self) -> None:
        """A segment has no volume, so reuse and unions are meaningless."""
        with self.assertRaises(ValueError):
            self._make(cls=AdaptiveSafePPOV2, compute_region_once=True)
        with self.assertRaises(ValueError):
            self._make(cls=AdaptiveSafePPOV2, region_update_mode="union")

    def test_learned_zonotope_still_rejects_directional_growth(self) -> None:
        """The relaxation is for segments only; 'zonotope' keeps its old guard."""
        with self.assertRaises(ValueError):
            self._make(safe_region_shape="zonotope", directional_rashomon_growth=True)

    # ------------------------------------------------------------- mechanics

    def test_region_requires_the_proposal_to_define_its_direction(self) -> None:
        model = self._make()
        with self.assertRaises(ValueError):
            model._compute_rashomon_around_last_safe()

    def test_certified_region_is_a_rank_one_zonotope_along_the_update(self) -> None:
        model = self._make()
        last_safe = [p.detach().clone() for p in model._last_safe_params]
        proposal = [p + 0.01 * th.randn_like(p) for p in last_safe]
        selected = model._compute_rashomon_around_last_safe(
            stop_target_params=proposal
        )
        self.assertIsNotNone(selected)
        region, _ = selected  # type: ignore[misc]
        self.assertIsInstance(region, ZonotopeRegion)
        self.assertEqual(int(region.generators.shape[0]), 1)

        alpha = model._last_segment_alpha
        self.assertIsNotNone(alpha)
        delta = flatten_tensors(
            [target - snapshot for target, snapshot in zip(proposal, last_safe)]
        )
        # The generator is alpha/2 * delta, i.e. parallel to the proposed update.
        generator = region.generators.reshape(-1).cpu()
        th.testing.assert_close(
            generator, (0.5 * float(alpha) * delta).cpu(), atol=1e-6, rtol=1e-5
        )

    def test_frozen_actor_is_restored_to_the_last_safe_params(self) -> None:
        model = self._make()
        last_safe = [p.detach().clone() for p in model._last_safe_params]
        proposal = [p + 0.05 * th.ones_like(p) for p in last_safe]
        model._compute_rashomon_around_last_safe(stop_target_params=proposal)
        for frozen, snapshot in zip(model._frozen_actor_params, last_safe):
            th.testing.assert_close(frozen.detach(), snapshot)

    def test_enforced_iterate_lands_on_the_certified_segment(self) -> None:
        """theta_{k+1} = theta_k + alpha_hat * delta, and is greedy-safe."""
        model = self._make()
        last_safe = [p.detach().clone() for p in model._last_safe_params]
        # Weights are zero and the safe action's bias is 5, so with one-hot
        # states the logits are exactly the bias. Pushing the unsafe action 0 to
        # a bias of 10 crosses the decision boundary halfway along the segment:
        # the proposal is unsafe, but alpha_star = 0.5 is available.
        with th.no_grad():
            bias = model._live_actor_params[-1]
            bias.data.copy_(th.tensor([10.0, 5.0, 0.0, 0.0], device=bias.device))
        candidate = [p.detach().clone() for p in model._live_actor_params]

        model._accept_or_project_candidate()

        alpha = float(model._last_segment_alpha or 0.0)
        self.assertGreater(alpha, 0.0)
        self.assertLess(alpha, 1.0)
        # alpha_star is exactly 0.5 here, so a tight certificate lands just below it.
        self.assertAlmostEqual(alpha, 0.5, delta=0.05)
        for enforced, snapshot, target in zip(
            model._live_actor_params, last_safe, candidate
        ):
            th.testing.assert_close(
                enforced.detach(),
                snapshot + alpha * (target - snapshot),
                atol=1e-5,
                rtol=1e-4,
            )
        self.assertTrue(model._verify_greedy_safe())

    def test_safe_candidate_is_accepted_without_a_segment_search(self) -> None:
        model = self._make()
        before = model.adaptive_diagnostics()["rashomon_computations"]
        model._accept_or_project_candidate()
        after = model.adaptive_diagnostics()
        self.assertEqual(after["rashomon_computations"], before)
        self.assertEqual(after["accepted_without_rashomon"], 1)

    def test_hopeless_candidate_reverts_to_the_last_safe_policy(self) -> None:
        model = self._make(segment_tolerance=1e-2)
        last_safe = [p.detach().clone() for p in model._last_safe_params]
        with th.no_grad():
            # Drive an unsafe action's logit far above the safe one.
            model._live_actor_params[-1].data.copy_(
                th.tensor([1e4, -1e4, 0.0, 0.0])[: model._live_actor_params[-1].numel()]
            )
        model._accept_or_project_candidate()
        for enforced, snapshot in zip(model._live_actor_params, last_safe):
            th.testing.assert_close(enforced.detach(), snapshot)
        self.assertEqual(model.adaptive_diagnostics()["fallback_reverts"], 1)

    def test_diagnostics_report_the_certified_step(self) -> None:
        model = self._make()
        last_safe = [p.detach().clone() for p in model._last_safe_params]
        proposal = [p + 0.02 * th.ones_like(p) for p in last_safe]
        model._compute_rashomon_around_last_safe(stop_target_params=proposal)
        diagnostics = model.adaptive_diagnostics()
        self.assertIsNotNone(diagnostics["segment_last_alpha"])
        self.assertIsNotNone(diagnostics["segment_mean_alpha"])
        self.assertGreaterEqual(int(diagnostics["segment_last_splits_used"]), 1)

    # ------------------------------------------------------------ end to end

    def test_v2_train_phase_run_stays_safe(self) -> None:
        model = self._make(
            cls=AdaptiveSafePPOV2,
            adaptive_granularity="train_phase",
            n_steps=64,
        )
        model.learn(total_timesteps=128)
        self.assertTrue(model._verify_greedy_safe())
        diagnostics = model.adaptive_diagnostics()
        self.assertEqual(diagnostics["accepted_unsafe"], 0)

    def test_v2_gradient_step_run_never_attaches_the_segment(self) -> None:
        """Enforcement-only: a rank-one region must not clamp the next step."""
        model = self._make(
            cls=AdaptiveSafePPOV2,
            adaptive_granularity="gradient_step",
            n_steps=64,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", message=".*GPU.*")
            model.learn(total_timesteps=128)
        self.assertTrue(model._verify_greedy_safe())
        attached = getattr(model.policy.optimizer, "_regions", None) or []
        self.assertFalse(
            any(isinstance(region, ZonotopeRegion) for region in attached),
            "the segment region was attached to the optimizer",
        )
        # ... but the optimizer must still hand the enforcement its proposal,
        # otherwise every step would revert for want of a direction.
        self.assertTrue(model.policy.optimizer._retain_proposals_only)
        self.assertIsNotNone(model.policy.optimizer.last_proposed_params)
        diagnostics = model.adaptive_diagnostics()
        self.assertEqual(diagnostics["accepted_unsafe"], 0)
        self.assertGreater(diagnostics["rashomon_computations"], 0)


if __name__ == "__main__":
    unittest.main()
