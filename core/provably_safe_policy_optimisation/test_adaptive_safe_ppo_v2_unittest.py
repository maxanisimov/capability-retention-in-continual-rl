"""Tests for AdaptiveSafePPOV2 projection-triggered region updates."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th

from provably_safe_policy_optimisation import AdaptiveSafePPOV2
from provably_safe_policy_optimisation import adaptive_safe_ppo as asp


def _only_action_safe(action: int, n_states: int = 16, n_actions: int = 4) -> np.ndarray:
    mask = np.zeros((n_states, n_actions), dtype=int)
    mask[:, action] = 1
    return mask


def _safe_base_state_dict(action: int = 1) -> dict[str, th.Tensor]:
    bias = th.zeros(4)
    bias[action] = 5.0
    return {"action_net.weight": th.zeros(4, 16), "action_net.bias": bias}


def _stub_engine_sequence(record: list, certificates: list[float], *, eps: float = 0.05):
    state = {"calls": 0}

    def stub(model, dataset, **kwargs):  # type: ignore[no-untyped-def]
        del dataset
        params = [p.detach().clone() for p in model.parameters()]
        index = min(state["calls"], len(certificates) - 1)
        state["calls"] += 1
        record.append({"params": params, "kwargs": kwargs})
        bounded = SimpleNamespace(
            param_l=[p - eps for p in params],
            param_u=[p + eps for p in params],
        )
        return SimpleNamespace(
            bounded_models=[bounded],
            certificates=[[SimpleNamespace(min_hard_acc=certificates[index])]],
        )

    return stub


class AdaptiveSafePPOV2Tests(unittest.TestCase):
    def _patch_engine(self, record: list, certificates: list[float] | None = None) -> None:
        original = asp._run_rashomon_engine
        asp._run_rashomon_engine = _stub_engine_sequence(record, certificates or [1.0])
        self.addCleanup(setattr, asp, "_run_rashomon_engine", original)

    def _make(self, **extra) -> AdaptiveSafePPOV2:
        env = gym.make("FrozenLake-v1")
        extra.setdefault("base_policy_state_dict", _safe_base_state_dict())
        extra.setdefault("n_steps", 32)
        extra.setdefault("batch_size", 16)
        extra.setdefault("n_epochs", 1)
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
        result = model.project_now()
        self.assertGreater(result.n_projected, 0)
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
