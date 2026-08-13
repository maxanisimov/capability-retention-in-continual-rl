"""Tests for AdaptiveSafePPO (adaptive verify-then-project updates) on FrozenLake-v1."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th

from provably_safe_policy_optimisation import AdaptiveSafePPO
from provably_safe_policy_optimisation import adaptive_safe_ppo as asp
from provably_safe_policy_optimisation.adaptive_safe_ppo import (
    calibrate_inverse_temperature,
    select_certified_box,
    shield_safe_behaviour_dataset,
)


def _only_action_safe(action: int, n_states: int = 16, n_actions: int = 4) -> np.ndarray:
    """A shield where only ``action`` is safe in every state."""
    mask = np.zeros((n_states, n_actions), dtype=int)
    mask[:, action] = 1
    return mask


def _safe_base_state_dict(action: int = 1) -> dict[str, th.Tensor]:
    """Actor weights (net_arch=[]) whose argmax is ``action`` in every state."""
    bias = th.zeros(4)
    bias[action] = 5.0
    return {"action_net.weight": th.zeros(4, 16), "action_net.bias": bias}


class _RecordActions(gym.Wrapper):
    """Records (state_before, action_executed) for every env step."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.records: list[tuple[int, int]] = []
        self._last: int = 0

    def reset(self, **kwargs):  # type: ignore[no-untyped-def]
        obs, info = self.env.reset(**kwargs)
        self._last = int(obs)
        return obs, info

    def step(self, action):  # type: ignore[no-untyped-def]
        self.records.append((self._last, int(action)))
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last = int(obs)
        return obs, reward, terminated, truncated, info


def _stub_engine(record: list, *, min_hard_acc: float = 1.0, eps: float = 0.05):
    """Fast deterministic stand-in for the Cooper Rashomon engine.

    Returns one box of half-width ``eps`` around the received model's current
    parameters (iterate k), certified at ``min_hard_acc``.
    """

    def stub(model, dataset, **kwargs):  # type: ignore[no-untyped-def]
        params = [p.detach().clone() for p in model.parameters()]
        record.append({"params": params, "kwargs": kwargs})
        bounded = SimpleNamespace(
            param_l=[p - eps for p in params],
            param_u=[p + eps for p in params],
        )
        return SimpleNamespace(
            bounded_models=[bounded],
            certificates=[[SimpleNamespace(min_hard_acc=min_hard_acc)]],
        )

    return stub


class AdaptiveSafePPOTests(unittest.TestCase):
    def _make(self, mask=None, recorder=None, **extra) -> AdaptiveSafePPO:
        env = recorder or gym.make("FrozenLake-v1")
        extra.setdefault("base_policy_state_dict", _safe_base_state_dict())
        extra.setdefault("n_steps", 32)
        extra.setdefault("batch_size", 16)
        extra.setdefault("n_epochs", 1)
        extra.setdefault("learning_rate", 1e-6)
        model = AdaptiveSafePPO(
            "MlpPolicy", env, shield=mask if mask is not None else _only_action_safe(1),
            seed=0, shield_seed=0, device="cpu", verbose=0,
            policy_kwargs={"net_arch": []}, **extra,
        )
        self.addCleanup(model.get_env().close)
        return model

    def _patch_engine(self, record: list, **stub_kwargs) -> None:
        original = asp._run_rashomon_engine
        asp._run_rashomon_engine = _stub_engine(record, **stub_kwargs)
        self.addCleanup(setattr, asp, "_run_rashomon_engine", original)

    def _force_unsafe_verifications(self, n_unsafe: int) -> None:
        """Make the first ``n_unsafe`` candidate greedy-rate checks report 0.0."""
        original = asp._greedy_safe_rate
        calls = {"n": 0}

        def fake(logits, safe_mask):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] <= n_unsafe:
                return 0.0
            return original(logits, safe_mask)

        asp._greedy_safe_rate = fake
        self.addCleanup(setattr, asp, "_greedy_safe_rate", original)

    # ------------------------------------------------------------ construction

    def test_base_policy_loaded_and_verified_at_init(self) -> None:
        model = self._make()
        params = dict(model.policy.named_parameters())
        self.assertTrue(th.equal(params["action_net.weight"].data, th.zeros(4, 16)))
        self.assertTrue(
            th.equal(params["action_net.bias"].data, _safe_base_state_dict()["action_net.bias"])
        )
        self.assertEqual(model._greedy_safe_rate_now(), 1.0)

    def test_unsafe_base_policy_raises(self) -> None:
        # Argmax = action 0 everywhere, but only action 1 is shield-safe.
        with self.assertRaises(ValueError):
            self._make(base_policy_state_dict=_safe_base_state_dict(action=0))

    def test_misaligned_base_state_dict_raises(self) -> None:
        missing = {"action_net.weight": th.zeros(4, 16)}
        with self.assertRaises(ValueError):
            self._make(base_policy_state_dict=missing)
        bad_shape = _safe_base_state_dict()
        bad_shape["action_net.bias"] = th.zeros(5)
        with self.assertRaises(ValueError):
            self._make(base_policy_state_dict=bad_shape)

    def test_shield_state_count_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make(mask=_only_action_safe(1, n_states=10))

    def test_invalid_granularity_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make(adaptive_granularity="per_epoch")

    def test_invalid_unsafe_update_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make(unsafe_update_strategy="teleport")

    def test_invalid_rashomon_surrogate_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make(rashomon_surrogate="unknown")

    def test_default_strategy_is_rashomon_project(self) -> None:
        model = self._make()
        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["unsafe_update_strategy"], "rashomon_project")

    # ----------------------------------------------------------- accept path

    def test_safe_candidates_accepted_without_rashomon(self) -> None:
        record: list = []
        self._patch_engine(record)
        model = self._make()
        model.learn(total_timesteps=32)
        diag = model.adaptive_diagnostics()
        # One train phase: n_epochs (1) * minibatches (32/16) = 2 candidates.
        self.assertEqual(diag["verifications_run"], 2)
        self.assertEqual(diag["candidates_verified_safe"], 2)
        self.assertEqual(diag["accepted_without_rashomon"], 2)
        self.assertEqual(diag["rashomon_computations"], 0)
        self.assertEqual(record, [])
        # Snapshot tracks the accepted iterate.
        for param, snapshot in zip(model._live_actor_params, model._last_safe_params):
            self.assertTrue(th.equal(param.data, snapshot))

    def test_gradient_step_granularity_verifies_every_optimizer_step(self) -> None:
        model = self._make(n_epochs=2)
        model.learn(total_timesteps=64)
        # Two train phases, each n_epochs (2) * minibatches (32/16) = 4 steps.
        self.assertEqual(model.adaptive_diagnostics()["verifications_run"], 8)
        self.assertIsNotNone(model.policy.optimizer.post_step_hook)

    def test_train_phase_granularity_verifies_once_per_train(self) -> None:
        model = self._make(adaptive_granularity="train_phase", n_epochs=2)
        model.learn(total_timesteps=64)
        self.assertEqual(model.adaptive_diagnostics()["verifications_run"], 2)
        self.assertIsNone(model.policy.optimizer.post_step_hook)

    # ---------------------------------------------------------- project path

    def test_unsafe_candidate_triggers_rashomon_and_projection(self) -> None:
        record: list = []
        eps = 0.05
        self._patch_engine(record, eps=eps)
        model = self._make(unsafe_update_strategy="rashomon_project")
        base_params = [p.detach().clone() for p in model._live_actor_params]
        self._force_unsafe_verifications(1)

        model.learn(total_timesteps=32)

        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["candidates_verified_unsafe"], 1)
        self.assertEqual(diag["rashomon_computations"], 1)
        self.assertEqual(diag["projections_applied"], 1)
        self.assertEqual(diag["fallback_reverts"], 0)
        self.assertIsNotNone(diag["last_projection"])
        # The engine received iterate k (the safe base), not the candidate.
        self.assertEqual(len(record), 1)
        for received, expected in zip(record[0]["params"], base_params):
            self.assertTrue(th.equal(received, expected))
        self.assertEqual(record[0]["kwargs"]["n_iters"], 100)
        self.assertEqual(record[0]["kwargs"]["checkpoint"], 10)
        # All safe states are certified by default (16 for this shield).
        self.assertEqual(record[0]["kwargs"]["certificate_samples"], 16)
        # The projected iterate lies inside the stub box around iterate k, and
        # the snapshot advanced to it.
        for param, base, snapshot in zip(
            model._live_actor_params, base_params, model._last_safe_params
        ):
            self.assertTrue(th.all(param.data >= base - eps - 1e-8))
            self.assertTrue(th.all(param.data <= base + eps + 1e-8))
            self.assertTrue(th.equal(param.data, snapshot))

    def test_logsumexp_surrogate_is_forwarded_to_engine_and_diagnostics(self) -> None:
        record: list = []
        self._patch_engine(record)
        model = self._make(rashomon_surrogate="logsumexp")
        self._force_unsafe_verifications(1)

        model.learn(total_timesteps=32)

        self.assertEqual(record[0]["kwargs"]["surrogate"], "logsumexp")
        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["rashomon_surrogate"], "logsumexp")
        self.assertEqual(diag["rashomon_resolved_surrogate"], "logsumexp")

    def test_no_certified_box_falls_back_to_revert(self) -> None:
        record: list = []
        self._patch_engine(record, min_hard_acc=0.5)
        # Single minibatch per phase so nothing moves after the revert.
        model = self._make(batch_size=32, unsafe_update_strategy="rashomon_project")
        base_params = [p.detach().clone() for p in model._live_actor_params]
        self._force_unsafe_verifications(1)

        model.learn(total_timesteps=32)

        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["rashomon_computations"], 1)
        self.assertEqual(diag["projections_applied"], 0)
        self.assertEqual(diag["fallback_reverts"], 1)
        # Candidate reverted to iterate k; snapshot unchanged.
        for param, base, snapshot in zip(
            model._live_actor_params, base_params, model._last_safe_params
        ):
            self.assertTrue(th.equal(param.data, base))
            self.assertTrue(th.equal(snapshot, base))

    def test_train_phase_projection(self) -> None:
        record: list = []
        self._patch_engine(record)
        model = self._make(
            adaptive_granularity="train_phase", unsafe_update_strategy="rashomon_project"
        )
        self._force_unsafe_verifications(1)
        model.learn(total_timesteps=32)
        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["verifications_run"], 1)
        self.assertEqual(diag["rashomon_computations"], 1)
        self.assertEqual(diag["projections_applied"], 1)

    # -------------------------------------------------- inherited behaviours

    def test_shielded_exploration_still_active(self) -> None:
        mask = _only_action_safe(1)
        recorder = _RecordActions(gym.make("FrozenLake-v1"))
        model = self._make(mask=mask, recorder=recorder)
        model.learn(total_timesteps=64)
        self.assertTrue(recorder.records)
        self.assertTrue(all(mask[s, a] == 1 for s, a in recorder.records))

    def test_no_static_bounds_attached(self) -> None:
        model = self._make()
        model.learn(total_timesteps=32)
        self.assertFalse(model.policy.optimizer.has_bounds)

    def test_mark_current_policy_safe(self) -> None:
        model = self._make()
        with th.no_grad():
            # Make the greedy action unsafe (argmax = action 0).
            dict(model.policy.named_parameters())["action_net.bias"].data.copy_(
                th.tensor([9.0, 0.0, 0.0, 0.0])
            )
        with self.assertRaises(ValueError):
            model.mark_current_policy_safe()
        model.mark_current_policy_safe(verify=False)
        for param, snapshot in zip(model._live_actor_params, model._last_safe_params):
            self.assertTrue(th.equal(param.data, snapshot))

    # ----------------------------------------------------- real-engine (slow)

    def test_real_engine_certifies_and_projects(self) -> None:
        try:
            import cooper  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            self.skipTest("cooper not installed")
        model = self._make(
            rashomon_n_iters=5, rashomon_batch_size=16, unsafe_update_strategy="rashomon_project"
        )
        self._force_unsafe_verifications(1)
        model.learn(total_timesteps=32)
        diag = model.adaptive_diagnostics()
        self.assertEqual(diag["rashomon_computations"], 1)
        # Either a certified box was found (projection) or the fallback fired;
        # with a greedy-safe base the initial box is feasible, so expect projection.
        self.assertEqual(diag["projections_applied"], 1)
        self.assertEqual(model._greedy_safe_rate_now(), 1.0)


class HelperTests(unittest.TestCase):
    def test_shield_safe_behaviour_dataset_excludes_no_safe_action_states(self) -> None:
        mask = np.array([[1, 0], [0, 0], [1, 1]], dtype=float)
        dataset, safe_ids = shield_safe_behaviour_dataset(mask)
        self.assertEqual(list(safe_ids), [0, 2])
        states, actions = dataset.tensors
        self.assertTrue(th.equal(states, th.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])))
        self.assertTrue(th.equal(actions, th.tensor([[1.0, 0.0], [1.0, 1.0]])))

    def test_shield_safe_behaviour_dataset_rejects_empty_shield(self) -> None:
        with self.assertRaises(ValueError):
            shield_safe_behaviour_dataset(np.zeros((3, 2)))

    def test_calibrate_inverse_temperature(self) -> None:
        logits = th.tensor([[0.5, 0.0], [0.0, 0.5]])
        masks = th.tensor([[1.0, 0.0], [0.0, 1.0]])
        # Single safe action -> threshold 0.5; small margin needs sharpening > 1.
        inverse_temp = calibrate_inverse_temperature(logits, masks)
        probs = th.softmax(logits * inverse_temp, dim=1)
        self.assertGreaterEqual(float((probs * masks).sum(dim=1).min()), 0.5)
        lse_inverse_temp = calibrate_inverse_temperature(
            logits, masks, surrogate="logsumexp",
        )
        self.assertGreaterEqual(lse_inverse_temp, 1)
        # A wrong-way policy can never clear the threshold.
        with self.assertRaises(ValueError):
            calibrate_inverse_temperature(-logits, masks, cap=10)

    def test_select_certified_box_picks_last_fully_certified(self) -> None:
        def box(value: float, certs: list[float]):
            return (
                SimpleNamespace(param_l=[th.tensor([value])], param_u=[th.tensor([value + 1.0])]),
                [SimpleNamespace(min_hard_acc=acc) for acc in certs],
            )

        boxes = [box(0.0, [1.0]), box(1.0, [1.0, 0.9]), box(2.0, [1.0, 1.0]), box(3.0, [0.8])]
        result = SimpleNamespace(
            bounded_models=[b[0] for b in boxes],
            certificates=[b[1] for b in boxes],
        )
        selected = select_certified_box(result)
        self.assertIsNotNone(selected)
        lower, upper, idx = selected
        self.assertEqual(idx, 2)  # last box whose *worst* group cert is 1.0
        self.assertTrue(th.equal(lower[0], th.tensor([2.0])))
        self.assertTrue(th.equal(upper[0], th.tensor([3.0])))

    def test_select_certified_box_returns_none_when_uncertified(self) -> None:
        result = SimpleNamespace(
            bounded_models=[SimpleNamespace(param_l=[th.zeros(1)], param_u=[th.ones(1)])],
            certificates=[[SimpleNamespace(min_hard_acc=0.99)]],
        )
        self.assertIsNone(select_certified_box(result))


if __name__ == "__main__":
    unittest.main()
