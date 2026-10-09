"""Tests for the PyTorch RL-SGF baseline."""

from __future__ import annotations

import math
import unittest
import warnings

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from safe_rl_baselines import RLSGF
from safe_rl_baselines.rl_sgf import (
    closed_form_step,
    discounted_returns_to_go,
    prop5_required_episodes,
    safety_margin,
    value_estimate_scale,
)


class TwoStepChain(gym.Env):
    """State 0 --a0--> state 1 + a0 --a1--> terminal.

    Reward and cost depend on (state, action); the exact value of a policy is a
    finite sum over the four trajectories, so exact gradients are available.
    """

    REWARD = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 2.0, (1, 1): 0.0, (2, 0): 0.5, (2, 1): 3.0}
    COST = {(0, 0): 0.0, (0, 1): 0.2, (1, 0): 0.0, (1, 1): 1.0, (2, 0): 0.0, (2, 1): 1.0}

    def __init__(self) -> None:
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self._state = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = 0
        return 0, {}

    def step(self, action):
        s, a = self._state, int(action)
        info = {"cost": self.COST[(s, a)]}
        if s == 0:
            self._state = 1 + a
            return self._state, self.REWARD[(s, a)], False, False, info
        return s, self.REWARD[(s, a)], True, False, info


class NoCostEnv(TwoStepChain):
    COST = {key: 0.0 for key in TwoStepChain.COST}


def _left_cost(obs, action, *rest) -> float:
    return 1.0 if int(action) == 0 else 0.0


def _exact_values(model: RLSGF, gamma: float, cost_gamma: float) -> tuple[th.Tensor, th.Tensor]:
    """Exact V0 (negated return) and V1 + cost_limit on TwoStepChain, differentiable in theta."""

    probs = th.softmax(model.actor(th.eye(3)), dim=-1)
    v_r = th.zeros(())
    v_c = th.zeros(())
    for a0 in (0, 1):
        s1 = 1 + a0
        for a1 in (0, 1):
            p = probs[0, a0] * probs[s1, a1]
            v_r = v_r + p * (TwoStepChain.REWARD[(0, a0)] + gamma * TwoStepChain.REWARD[(s1, a1)])
            v_c = v_c + p * (TwoStepChain.COST[(0, a0)] + cost_gamma * TwoStepChain.COST[(s1, a1)])
    return -v_r, v_c


def _qcqp_reference(g0: np.ndarray, g1: np.ndarray, v1: float, h: float, alpha: float) -> np.ndarray:
    """Independent solution of QCQP (3) by geometry, not via Lemma 3.

    The objective is (1/2h)||d + h g0||^2 + const and the constraint set is the ball
    with centre -h g1 and radius h sqrt(||g1||^2 - 2 alpha v1), so the minimiser is the
    Euclidean projection of the gradient step -h g0 onto that ball.
    """

    centre = -h * g1
    radius = h * math.sqrt(g1 @ g1 - 2 * alpha * v1)
    offset = -h * g0 - centre
    dist = float(np.linalg.norm(offset))
    return -h * g0 if dist <= radius else centre + offset * (radius / dist)


class ClosedFormStepTests(unittest.TestCase):
    H, ALPHA = 0.5, 1.0

    def _check_against_solver(self, g0, g1, v1, expected_case):
        d, case, _ = closed_form_step(th.tensor(g0), th.tensor(g1), v1, step_size=self.H, alpha=self.ALPHA)
        self.assertEqual(case, expected_case)
        d_ref = _qcqp_reference(np.asarray(g0, dtype=float), np.asarray(g1, dtype=float), v1, self.H, self.ALPHA)
        np.testing.assert_allclose(d.numpy(), d_ref, atol=1e-5)

    def test_unconstrained_case_matches_solver(self) -> None:
        # Far inside the safe set: the plain gradient step already satisfies the ball constraint.
        self._check_against_solver([0.3, -0.2, 0.1], [0.1, 0.0, 0.2], -5.0, "unconstrained")

    def test_constrained_case_matches_solver(self) -> None:
        # Reward gradient points straight up the cost: the constraint is active.
        rng = np.random.default_rng(0)
        for _ in range(20):
            g1 = rng.normal(size=4)
            g0 = -2.0 * g1 + 0.3 * rng.normal(size=4)
            v1 = -0.05 * rng.random()
            self._check_against_solver(g0.tolist(), g1.tolist(), v1, "constrained")

    def test_constrained_case_from_slightly_unsafe_policy(self) -> None:
        g1 = np.array([1.0, 0.5])
        g0 = np.array([-1.0, 0.2])
        self._check_against_solver(g0.tolist(), g1.tolist(), 0.05, "constrained")

    def test_constrained_step_lands_on_constraint_boundary(self) -> None:
        g0, g1, v1 = th.tensor([-1.0, 0.3]), th.tensor([1.0, 0.0]), -0.1
        d, case, u = closed_form_step(g0, g1, v1, step_size=self.H, alpha=self.ALPHA)
        self.assertEqual(case, "constrained")
        self.assertGreater(u, 0.0)
        lhs = self.ALPHA * self.H * v1 + float(g1 @ d) + float(d @ d) / (2 * self.H)
        self.assertAlmostEqual(lhs, 0.0, places=6)

    def test_infeasible_falls_back_to_least_violation_step(self) -> None:
        g0, g1, v1 = th.tensor([1.0, 0.0]), th.tensor([0.1, 0.0]), 2.0   # A = 0.01 - 4 < 0
        d, case, u = closed_form_step(g0, g1, v1, step_size=self.H, alpha=self.ALPHA)
        self.assertEqual(case, "infeasible")
        self.assertTrue(math.isinf(u))
        self.assertTrue(th.allclose(d, -self.H * g1))

    def test_zero_cost_signal_at_zero_budget_is_stuck(self) -> None:
        d, case, _ = closed_form_step(th.tensor([1.0, -2.0]), th.zeros(2), 0.0, step_size=self.H, alpha=self.ALPHA)
        self.assertEqual(case, "stuck")
        self.assertEqual(float(th.linalg.vector_norm(d)), 0.0)

    def test_boundary_with_nonzero_cost_gradient_takes_constrained_step(self) -> None:
        # V1 == 0 but grad V1 != 0: A > 0, so the policy can still move along the boundary.
        d, case, _ = closed_form_step(th.tensor([-1.0, -1.0]), th.tensor([1.0, 0.0]), 0.0,
                                      step_size=self.H, alpha=self.ALPHA)
        self.assertEqual(case, "constrained")
        self.assertGreater(float(th.linalg.vector_norm(d)), 0.0)


class ExactAnytimeSafetyTests(unittest.TestCase):
    """Lemma 1 with exact quantities: iterates from a feasible point stay feasible and reach a KKT point."""

    def test_feasible_iterates_stay_feasible_and_converge(self) -> None:
        center, radius, lip = th.tensor([0.0, 0.0]), 1.0, 2.0       # V1 = (L/2)||x||^2 - r, L1 = 2
        target = th.tensor([3.0, 1.0])                               # V0 = 0.5||x - target||^2 (minimum infeasible)
        h, alpha = 0.4, 1.0                                          # h < 1/L0 = 1, h < 1/L1 = 0.5
        x = th.tensor([0.2, -0.5])
        for _ in range(300):
            v1 = 0.5 * lip * float((x - center) @ (x - center)) - radius
            self.assertLessEqual(v1, 1e-9)
            d, _, _ = closed_form_step(x - target, lip * (x - center), v1, step_size=h, alpha=alpha)
            x = x + d
        self.assertLess(float(th.linalg.vector_norm(d)), 1e-6)
        # KKT point of min 0.5||x-t||^2 s.t. ||x||^2 <= 1: the projection of t onto the unit ball.
        self.assertTrue(th.allclose(x, target / th.linalg.vector_norm(target), atol=1e-4))


class EstimatorTests(unittest.TestCase):
    GAMMA, COST_GAMMA = 0.9, 0.8

    def _model(self, env, **extra) -> RLSGF:
        return RLSGF(env, net_arch=(8,), gamma=self.GAMMA, cost_gamma=self.COST_GAMMA, cost_limit=0.1,
                     seed=0, device="cpu", **extra)

    def test_discounted_returns_to_go(self) -> None:
        np.testing.assert_allclose(discounted_returns_to_go(np.array([1.0, 2.0, 4.0]), 0.5), [3.0, 4.0, 4.0])

    def _check_unbiased(self, baseline: str) -> None:
        model = self._model(TwoStepChain(), baseline=baseline)
        if baseline == "critic":
            # A deliberately wrong (but batch-independent) baseline must not bias the gradient.
            with th.no_grad():
                for critic in (model.reward_critic, model.cost_critic):
                    critic[-1].bias.fill_(0.7)
        exact_v0, exact_c = _exact_values(model, self.GAMMA, self.COST_GAMMA)
        exact_g0 = model._flat_grad(exact_v0, retain_graph=True)
        exact_g1 = model._flat_grad(exact_c)
        exact_v0, exact_c = exact_v0.detach(), exact_c.detach()
        est = model.estimate(model.collect_episodes(20_000))
        self.assertAlmostEqual(est["v0_hat"], float(exact_v0), delta=0.05)
        self.assertAlmostEqual(est["v1_hat"], float(exact_c) - 0.1, delta=0.02)
        for got, want in ((est["grad_v0"], exact_g0), (est["grad_v1"], exact_g1)):
            err = float(th.linalg.vector_norm(got - want))
            self.assertLess(err, 0.03 * max(1.0, float(th.linalg.vector_norm(want))))

    def test_reinforce_estimates_are_unbiased(self) -> None:
        self._check_unbiased("none")

    def test_critic_baseline_keeps_estimates_unbiased(self) -> None:
        self._check_unbiased("critic")

    def test_training_records_and_timesteps(self) -> None:
        model = self._model(TwoStepChain(), episodes_per_iter=7)
        model.learn(total_timesteps=1)
        self.assertEqual(len(model.training_episodes), 7)
        self.assertEqual(model.num_timesteps, 14)
        self.assertIn(model.last_stats["case"], {"unconstrained", "constrained", "cost_descent", "infeasible"})
        self.assertIn("training_violation_percentage", model.last_stats)


class StuckAtZeroBudgetTests(unittest.TestCase):
    def test_zero_budget_without_cost_never_moves(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = RLSGF(NoCostEnv(), net_arch=(8,), cost_limit=0.0, episodes_per_iter=5, seed=0, device="cpu")
        self.assertTrue(any("cost_limit <= 0" in str(w.message) for w in caught))
        before = model._flat_params().clone()
        model.learn(total_timesteps=30)
        self.assertTrue(th.equal(model._flat_params(), before))
        self.assertEqual(set(model.case_counts), {"stuck"})


class ValidationAndBoundsTests(unittest.TestCase):
    def test_rejects_step_size_times_alpha_at_least_one(self) -> None:
        with self.assertRaises(ValueError):
            RLSGF(TwoStepChain(), step_size=0.5, alpha=2.0, cost_limit=0.1, device="cpu")

    def test_rejects_unknown_baseline(self) -> None:
        with self.assertRaises(ValueError):
            RLSGF(TwoStepChain(), baseline="gae", cost_limit=0.1, device="cpu")

    def test_prop5_bound(self) -> None:
        self.assertTrue(math.isinf(prop5_required_episodes(1.0, 1.0, 10, 0.0, 0.05)))
        loose = prop5_required_episodes(1.0, 1.0, 10, 0.5, 0.05)
        tight = prop5_required_episodes(1.0, 1.0, 10, 0.1, 0.05)
        self.assertAlmostEqual(tight / loose, 25.0)
        self.assertAlmostEqual(value_estimate_scale(1.0, 0.5, 2), 1.75)

    def test_safety_margin_shrinks_with_cost(self) -> None:
        safe = safety_margin(-0.5, 0.1, step_size=0.2, alpha=1.0, cost_lipschitz=1.0)
        unsafe = safety_margin(0.5, 0.1, step_size=0.2, alpha=1.0, cost_lipschitz=1.0)
        self.assertGreater(safe, 0.0)
        self.assertLess(unsafe, 0.0)


class CartPoleSmokeTests(unittest.TestCase):
    def test_learns_and_reduces_cost_on_cartpole(self) -> None:
        model = RLSGF(
            gym.make("CartPole-v1"), cost_fn=_left_cost, cost_limit=5.0, gamma=0.99, cost_gamma=0.99,
            step_size=0.05, alpha=1.0, episodes_per_iter=10, seed=0, device="cpu", net_arch=(32,),
        )
        before = model._flat_params().clone()
        model.learn(total_timesteps=3_000)
        self.assertFalse(th.equal(model._flat_params(), before))
        for key in ("v0_hat", "v1_hat", "case", "step_norm", "case_counts", "training_violation_percentage"):
            self.assertIn(key, model.last_stats)
        action, _ = model.predict(np.zeros(4, dtype=np.float32), deterministic=True)
        self.assertIn(int(action), (0, 1))
        for attr in ("actor", "reward_critic", "cost_critic"):
            self.assertTrue(hasattr(model, attr))


if __name__ == "__main__":
    unittest.main()
