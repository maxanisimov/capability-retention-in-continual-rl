"""Reinforcement Learning-based Safe Gradient Flow (RL-SGF): an anytime safe-RL baseline.

PyTorch implementation of RL-SGF (Mestres, Marzabal & Cortes, 2025, "Anytime Safe
Reinforcement Learning", L4DC, PMLR 283:221-232). The constrained MDP is written as

    minimise   V0(theta) = -E[sum_t gamma^t r_t]                      (negated return)
    s.t.       V1(theta) =  E[sum_t gamma_c^t c_t] - cost_limit <= 0  (cost budget)

Each iteration collects ``episodes_per_iter`` complete on-policy episodes, forms the
Monte-Carlo / REINFORCE estimates of Lemma 2 (``V1_hat``, ``grad V0_hat``,
``grad V1_hat``) and takes the step ``theta <- p_hat(theta)``, where

    p_hat(theta) = argmin_y  gV0 . (y - theta) + 1/(2h) ||y - theta||^2
                   s.t.      alpha h V1 + gV1 . (y - theta) + 1/(2h) ||y - theta||^2 <= 0

is a discretisation of the Safe Gradient Flow (a "moving balls" step). The constraint
is a quadratic upper bound on V1 when h < 1/L1, so with exact quantities a feasible
policy stays feasible (Lemma 1); with estimates this holds with high probability for
enough episodes (Proposition 5). The QCQP has the closed form of Lemma 3, so no
solver is needed: with A = ||gV1||^2 - 2 alpha V1 and
C = 2 gV1.gV0 - ||gV0||^2 - 2 alpha V1,

    A > 0, C >= 0 :  y = theta - h gV0                                ("unconstrained")
    A > 0, C <  0 :  y = theta - h/(1+u) (gV0 + u gV1),
                     u = ||gV1 - gV0|| / sqrt(A) - 1                  ("constrained")
    A <= 0        :  y = theta - h gV1                                ("cost_descent")

``A == 0`` is the paper's third case (the feasible set is the single point
``theta - h gV1``); ``A < 0`` means the QCQP is infeasible (the current policy is
estimated to be too unsafe), and the same pure cost-descent step is used as the
least-violating point. There is no dual variable to initialise or tune.

Practical notes
---------------
* **Use a positive cost budget.** With ``cost_limit <= 0`` and a policy that incurs
  no cost on the sampled episodes, ``V1_hat == 0`` and ``grad V1_hat == 0``, so
  ``A == 0`` and the step is exactly zero: RL-SGF never moves (Slater's condition
  fails). These iterations are reported as ``case == "stuck"``.
* ``step_size * alpha`` must be < 1 (Lemma 1). The guarantee additionally needs
  ``step_size < 1/L0, 1/L1``; for an MLP those constants are unknown, so the step
  size is a hyperparameter and the anytime guarantee is not formal.
* The constraint uses the *discounted* episode cost (``cost_gamma``); evaluation
  elsewhere in the repo counts undiscounted episode cost.
* The baseline in the gradient estimator is ``G_t - b(s_t)`` (standard REINFORCE);
  the paper's displayed estimator subtracts ``b(s_t)`` inside the reward-to-go sum,
  which is also unbiased but appears to be a typo. ``baseline="none"`` (b = 0)
  matches the paper's experiments; ``baseline="critic"`` uses the reward / cost
  critics, fit by regression on Monte-Carlo returns *after* each gradient estimate
  so the baseline is independent of the current batch.

This subclasses :class:`PPOLagrangian` only to reuse infrastructure (actor + critics,
observation preprocessing, the cost source, ``predict`` and the training-episode
records); the Lagrange multiplier and PPO hyperparameters are unused.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np
import torch as th

from safe_rl_baselines.cpo import CPO
from safe_rl_baselines.ppo_lagrangian import PPOLagrangian

BASELINE_MODES = ("none", "critic")


def discounted_returns_to_go(values: np.ndarray, gamma: float) -> np.ndarray:
    """``out[t] = sum_{t' >= t} gamma^(t'-t) values[t']`` for one episode."""

    out = np.zeros(len(values), dtype=np.float64)
    running = 0.0
    for t in reversed(range(len(values))):
        running = float(values[t]) + gamma * running
        out[t] = running
    return out


def closed_form_step(
    grad_v0: th.Tensor,
    grad_v1: th.Tensor,
    v1: float,
    *,
    step_size: float,
    alpha: float,
    tol: float = 1e-12,
) -> tuple[th.Tensor, str, float]:
    """Solve the RL-SGF QCQP (paper eq. 3) in closed form (Lemma 3).

    Returns ``(d, case, u)`` with ``d = y* - theta`` the parameter displacement,
    ``case`` one of ``"unconstrained"``, ``"constrained"``, ``"cost_descent"``,
    ``"stuck"`` (``A == 0`` with a zero cost gradient, so ``d == 0``) or
    ``"infeasible"`` (``A < 0``; least-violation cost-descent step), and ``u`` the
    constraint multiplier (``inf`` for the cost-descent cases).
    """

    h = float(step_size)
    g0 = grad_v0.double()
    g1 = grad_v1.double()
    g1_sq = float(th.dot(g1, g1))
    a = g1_sq - 2.0 * alpha * float(v1)
    if a <= tol:
        d = (-h * g1).to(grad_v0.dtype)
        if a < -tol:
            return d, "infeasible", math.inf
        return d, ("stuck" if g1_sq <= tol else "cost_descent"), math.inf
    c = 2.0 * float(th.dot(g1, g0)) - float(th.dot(g0, g0)) - 2.0 * alpha * float(v1)
    if c >= 0.0:
        return (-h * g0).to(grad_v0.dtype), "unconstrained", 0.0
    # Active constraint: A u^2 + 2A u + C = 0, positive root u = ||g1 - g0|| / sqrt(A) - 1.
    one_plus_u = float(th.linalg.vector_norm(g1 - g0)) / math.sqrt(a)
    u = one_plus_u - 1.0
    d = -(h / one_plus_u) * (g0 + u * g1)
    return d.to(grad_v0.dtype), "constrained", u


def safety_margin(
    v1: float,
    step_norm: float,
    *,
    step_size: float,
    alpha: float,
    cost_lipschitz: float,
) -> float:
    """Estimation-error margin of Proposition 5.

    If ``|V1 - V1_hat| <= M`` and ``||grad V1 - grad V1_hat|| <= M`` the next iterate
    is safe. Covers both cases of the proposition: for ``V1_hat <= 0`` this is
    ``M_hat_i``; for ``V1_hat > 0`` it is the supremum of admissible ``nu`` in (4).
    A non-positive value means no guarantee is available for this step.
    """

    slack = 0.5 * (1.0 / step_size - cost_lipschitz) * step_norm ** 2 - (1.0 - alpha * step_size) * v1
    return slack / (1.0 + step_norm)


def value_estimate_scale(cost_bound: float, gamma: float, horizon: int) -> float:
    """``sigma_tilde`` of Lemma 2: range of one episode's discounted cost."""

    return cost_bound * (1.0 - gamma ** (horizon + 1)) / (1.0 - gamma) if gamma != 1.0 else cost_bound * (horizon + 1)


def gradient_estimate_scale(
    grad_log_bound: float, cost_bound: float, baseline_bound: float, gamma: float, horizon: int
) -> float:
    """``sigma_bar`` of Lemma 2 (per-coordinate scale of the REINFORCE estimator)."""

    total = 0.0
    for t in range(horizon + 1):
        inner = sum(cost_bound * gamma ** (tp - t) + baseline_bound for tp in range(t, horizon + 1))
        total += gamma ** t * inner
    return grad_log_bound * total


def prop5_required_episodes(
    sigma_tilde: float, sigma_bar: float, n_params: int, margin: float, delta: float
) -> float:
    """Lower bound on the episodes per iteration from Proposition 5.

    With more episodes than this, ``P(V1(next) <= 0) >= 1 - 2 delta``. Returns
    ``inf`` when ``margin <= 0`` (no number of episodes certifies the step).
    """

    if margin <= 0.0:
        return math.inf
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")
    value_term = -2.0 * sigma_tilde ** 2 * math.log(delta) / margin ** 2
    grad_term = -2.0 * n_params * sigma_bar ** 2 * math.log(delta / n_params) / margin ** 2
    return max(value_term, grad_term)


class RLSGF(PPOLagrangian):
    """RL-SGF: Monte-Carlo safe gradient flow with a closed-form QCQP step."""

    # Flat policy-parameter helpers are shared with CPO (same actor / log_std layout).
    _policy_params = CPO._policy_params
    _flat_params = CPO._flat_params
    _set_flat_params = CPO._set_flat_params
    _flat_grad = CPO._flat_grad

    def __init__(
        self,
        env: Any,
        *,
        step_size: float = 0.1,
        alpha: float = 1.0,
        episodes_per_iter: int = 100,
        baseline: str = "none",
        n_critic_updates: int = 20,
        critic_lr: float = 1e-3,
        cost_lipschitz: float | None = None,
        max_episode_length: int | None = None,
        **shared: Any,
    ) -> None:
        super().__init__(env, **shared)
        if step_size <= 0.0 or alpha <= 0.0:
            raise ValueError("step_size and alpha must be positive.")
        if step_size * alpha >= 1.0:
            raise ValueError(f"RL-SGF requires step_size * alpha < 1 (got {step_size * alpha:g}).")
        if baseline not in BASELINE_MODES:
            raise ValueError(f"baseline must be one of {BASELINE_MODES}, got {baseline!r}.")
        if int(episodes_per_iter) < 1:
            raise ValueError("episodes_per_iter must be >= 1.")
        if self.cost_limit <= 0.0:
            warnings.warn(
                "RL-SGF with cost_limit <= 0: a policy that incurs no sampled cost has "
                "V1_hat = 0 and grad V1_hat = 0, so the update is exactly zero (Slater's "
                "condition fails). Use a positive cost budget.",
                stacklevel=2,
            )
        self.step_size = float(step_size)
        self.alpha = float(alpha)
        self.episodes_per_iter = int(episodes_per_iter)
        self.baseline = baseline
        self.n_critic_updates = int(n_critic_updates)
        self.cost_lipschitz = cost_lipschitz
        self.max_episode_length = max_episode_length
        # The policy is updated by the closed-form step; only the critics (used as
        # optional baselines) are trained by gradient descent.
        self.critic_optimizer = th.optim.Adam(
            list(self.reward_critic.parameters()) + list(self.cost_critic.parameters()),
            lr=critic_lr,
        )
        self.case_counts: dict[str, int] = {}
        self._episodes: list[dict[str, np.ndarray]] = []

    # --- data collection -------------------------------------------------
    def collect_episodes(self, n_episodes: int) -> list[dict[str, np.ndarray]]:
        """Run ``n_episodes`` complete on-policy episodes from ``env.reset()``.

        Episodes end on ``terminated`` / ``truncated`` (the env's time limit is the
        horizon T) or after ``max_episode_length`` steps. Truncation is not
        bootstrapped: RL-SGF optimises the finite-horizon objective.
        """

        episodes: list[dict[str, np.ndarray]] = []
        self._last_rollout_training_episodes = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            obs_rows: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            costs: list[float] = []
            while True:
                obs_t = self._preprocess(obs)
                with th.no_grad():
                    action = self._distribution(obs_t).sample()
                action_env = int(action.item()) if self._discrete_actions else action.cpu().numpy()[0]
                next_obs, reward, terminated, truncated, info = self.env.step(action_env)
                cost = self._step_cost(obs, action_env, reward, next_obs, terminated, truncated, info)
                obs_rows.append(obs_t.cpu().numpy()[0])
                actions.append(action.cpu().numpy().reshape(self._act_shape))
                rewards.append(float(reward))
                costs.append(float(cost))
                self.num_timesteps += 1
                if self._exploration_action_callback is not None:
                    self._exploration_action_callback(
                        timestep=int(self.num_timesteps), obs=obs, action=action_env
                    )
                obs = next_obs
                if terminated or truncated:
                    break
                if self.max_episode_length is not None and len(rewards) >= self.max_episode_length:
                    break
            episode = {
                "obs": np.asarray(obs_rows, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.float32),
                "rewards": np.asarray(rewards, dtype=np.float64),
                "costs": np.asarray(costs, dtype=np.float64),
            }
            episodes.append(episode)
            ep_cost = float(episode["costs"].sum())
            record = {
                "episode": self._training_episode_index,
                "end_timestep": self.num_timesteps,
                "reward": float(episode["rewards"].sum()),
                "cost": ep_cost,
                "length": int(len(rewards)),
                "violated": bool(ep_cost > self.cost_limit),
            }
            self.training_episodes.append(record)
            self._last_rollout_training_episodes.append(record)
            self._training_episode_index += 1
        self._episodes = episodes
        return episodes

    # --- estimates (Lemma 2) ---------------------------------------------
    def _batch(self, episodes: list[dict[str, np.ndarray]]) -> dict[str, th.Tensor | float]:
        obs = np.concatenate([ep["obs"] for ep in episodes])
        actions = np.concatenate([ep["actions"] for ep in episodes])
        disc_r, disc_c, ret_r, ret_c, v0_terms, v1_terms = [], [], [], [], [], []
        for ep in episodes:
            n = len(ep["rewards"])
            ret_r.append(discounted_returns_to_go(ep["rewards"], self.gamma))
            ret_c.append(discounted_returns_to_go(ep["costs"], self.cost_gamma))
            disc_r.append(self.gamma ** np.arange(n))
            disc_c.append(self.cost_gamma ** np.arange(n))
            v0_terms.append(-ret_r[-1][0] if n else 0.0)
            v1_terms.append(ret_c[-1][0] if n else 0.0)
        as_t = lambda x: th.as_tensor(np.concatenate(x), dtype=th.float32, device=self.device)  # noqa: E731
        obs_t = th.as_tensor(obs, dtype=th.float32, device=self.device)
        if self._discrete_actions:
            act_t = th.as_tensor(actions.reshape(-1), dtype=th.long, device=self.device)
        else:
            act_t = th.as_tensor(actions, dtype=th.float32, device=self.device)
        return {
            "obs": obs_t,
            "actions": act_t,
            "ret_r": as_t(ret_r),
            "ret_c": as_t(ret_c),
            "disc_r": as_t(disc_r),
            "disc_c": as_t(disc_c),
            "v0_hat": float(np.mean(v0_terms)),
            "v1_hat": float(np.mean(v1_terms)) - float(self.cost_limit),
            "mean_episode_cost": float(np.mean([ep["costs"].sum() for ep in episodes])),
            "mean_episode_reward": float(np.mean([ep["rewards"].sum() for ep in episodes])),
        }

    def estimate(self, episodes: list[dict[str, np.ndarray]]) -> dict[str, Any]:
        """``V0_hat``, ``V1_hat`` and the REINFORCE gradients ``grad V0_hat``, ``grad V1_hat``."""

        batch = self._batch(episodes)
        obs, actions = batch["obs"], batch["actions"]
        with th.no_grad():
            if self.baseline == "critic":
                b_r = self.reward_critic(obs).flatten()
                b_c = self.cost_critic(obs).flatten()
            else:
                b_r = th.zeros_like(batch["ret_r"])
                b_c = th.zeros_like(batch["ret_c"])
            w0 = -batch["disc_r"] * (batch["ret_r"] - b_r)   # V0 = -return
            w1 = batch["disc_c"] * (batch["ret_c"] - b_c)
        logp = self._action_log_prob(self._distribution(obs), actions)
        n_ep = float(len(episodes))
        grad_v0 = self._flat_grad((logp * w0).sum() / n_ep, retain_graph=True).detach()
        grad_v1 = self._flat_grad((logp * w1).sum() / n_ep).detach()
        return {**batch, "grad_v0": grad_v0, "grad_v1": grad_v1}

    def _fit_critics(self, batch: dict[str, Any]) -> float:
        loss_val = 0.0
        for _ in range(self.n_critic_updates):
            loss = ((batch["ret_r"] - self.reward_critic(batch["obs"]).flatten()) ** 2).mean() + (
                (batch["ret_c"] - self.cost_critic(batch["obs"]).flatten()) ** 2
            ).mean()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            loss_val = float(loss.item())
        return loss_val

    # --- update ----------------------------------------------------------
    def train(self) -> None:
        est = self.estimate(self._episodes)
        d, case, u = closed_form_step(
            est["grad_v0"], est["grad_v1"], est["v1_hat"], step_size=self.step_size, alpha=self.alpha
        )
        self._set_flat_params(self._flat_params() + d)
        self.case_counts[case] = self.case_counts.get(case, 0) + 1
        value_loss = self._fit_critics(est) if self.baseline == "critic" else 0.0
        step_norm = float(th.linalg.vector_norm(d))
        self.last_stats = {
            "v0_hat": est["v0_hat"],
            "v1_hat": est["v1_hat"],
            "grad_v0_norm": float(th.linalg.vector_norm(est["grad_v0"])),
            "grad_v1_norm": float(th.linalg.vector_norm(est["grad_v1"])),
            "case": case,
            "u": float(u),
            "step_norm": step_norm,
            "mean_episode_cost": est["mean_episode_cost"],
            "mean_episode_reward": est["mean_episode_reward"],
            "episodes_per_iter": float(len(self._episodes)),
            "value_loss": value_loss,
            "case_counts": dict(self.case_counts),
        }
        if self.cost_lipschitz is not None:
            self.last_stats["prop5_margin"] = safety_margin(
                est["v1_hat"], step_norm, step_size=self.step_size, alpha=self.alpha,
                cost_lipschitz=float(self.cost_lipschitz),
            )

    def learn(self, total_timesteps: int, callback: Callable[["RLSGF"], bool] | None = None) -> "RLSGF":
        while self.num_timesteps < total_timesteps:
            self.collect_episodes(self.episodes_per_iter)
            self.train()
            self.last_stats.update(self._training_violation_stats())
            if callback is not None and not callback(self):
                break
        return self
