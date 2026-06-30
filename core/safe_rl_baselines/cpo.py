"""Constrained Policy Optimization (CPO): a PyTorch safe-RL baseline.

PyTorch implementation of CPO (Achiam, Held, Tamar & Abbeel, 2017,
"Constrained Policy Optimization", arXiv:1705.10528), ported from MASA's JAX
``masa.algorithms.on_policy.cpo`` (equivalent to OpenAI safety-starter-agents).

CPO is a **standalone trust-region** method in the TRPO family -- it does NOT wrap
PPO/SAC. Each update solves, to first/second order around the current policy:

    maximise   g . (theta - theta_k)                         (reward surrogate)
    s.t.       c + b . (theta - theta_k) <= 0                 (cost surrogate)
               1/2 (theta - theta_k)^T H (theta - theta_k) <= delta   (KL trust region)

where ``H`` is the Fisher information (KL Hessian), ``g``/``b`` are the reward/cost
surrogate gradients, and ``c = J_C(theta_k) - cost_limit``. It is solved analytically
via the Lagrangian dual using conjugate-gradient Fisher-vector products, then a
backtracking line search enforces the surrogate cost constraint and the KL trust
region. When the current policy is infeasible (``c > 0`` and the trust region cannot
satisfy the constraint) CPO takes a pure cost-reduction *recovery* step.

This subclasses :class:`PPOLagrangian` only to reuse infrastructure (actor + reward
critic + cost critic, observation preprocessing, rollout collection, the cost source,
and GAE); the Lagrange multiplier is unused.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from safe_rl_baselines.ppo_lagrangian import PPOLagrangian, _gae


class CPO(PPOLagrangian):
    """Constrained Policy Optimization (trust-region, TRPO-family)."""

    def __init__(
        self,
        env: Any,
        *,
        target_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        line_search_steps: int = 10,
        line_search_decay: float = 0.8,
        n_critic_updates: int = 80,
        critic_lr: float = 1e-3,
        fvp_sample_freq: int = 1,
        **shared: Any,
    ) -> None:
        super().__init__(env, **shared)
        self.target_kl = target_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_steps = line_search_steps
        self.line_search_decay = line_search_decay
        self.n_critic_updates = n_critic_updates
        self.fvp_sample_freq = max(1, int(fvp_sample_freq))
        # CPO updates the actor manually; critics are fit by regression here. The
        # inherited joint self.optimizer is unused.
        self.critic_optimizer = th.optim.Adam(
            list(self.reward_critic.parameters()) + list(self.cost_critic.parameters()),
            lr=critic_lr,
        )
        self._mean_ep_cost = 0.0

    # --- policy parameter vector helpers ---------------------------------
    def _policy_params(self) -> list[th.nn.Parameter]:
        params = list(self.actor.parameters())
        if not self._discrete_actions:
            params.append(self.log_std)
        return params

    def _flat_params(self) -> th.Tensor:
        return th.cat([p.data.reshape(-1) for p in self._policy_params()])

    def _set_flat_params(self, flat: th.Tensor) -> None:
        i = 0
        for p in self._policy_params():
            n = p.numel()
            p.data.copy_(flat[i:i + n].view_as(p))
            i += n

    def _flat_grad(self, loss: th.Tensor, *, create_graph: bool = False, retain_graph: bool | None = None) -> th.Tensor:
        # retain_graph=None lets autograd default to create_graph (so create_graph=True
        # retains the graph for the second-order backward in the Fisher-vector product).
        grads = th.autograd.grad(loss, self._policy_params(), create_graph=create_graph, retain_graph=retain_graph)
        return th.cat([g.contiguous().reshape(-1) for g in grads])

    # --- KL / Fisher-vector product / CG ---------------------------------
    def _mean_kl(self, obs: th.Tensor, old: dict[str, th.Tensor]) -> th.Tensor:
        """Mean KL( old || current_policy(obs) ); ``old`` holds detached policy outputs."""
        out = self.actor(obs)
        if self._discrete_actions:
            log_p_old = F.log_softmax(old["logits"], dim=-1)
            log_p_new = F.log_softmax(out, dim=-1)
            p_old = log_p_old.exp()
            return (p_old * (log_p_old - log_p_new)).sum(-1).mean()
        mean_old, log_std_old = old["mean"], old["log_std"]
        std_old, std_new = log_std_old.exp(), self.log_std.exp()
        kl = (
            self.log_std - log_std_old
            + (std_old.pow(2) + (mean_old - out).pow(2)) / (2.0 * std_new.pow(2))
            - 0.5
        )
        return kl.sum(-1).mean()

    def _old_outputs(self, obs: th.Tensor) -> dict[str, th.Tensor]:
        with th.no_grad():
            out = self.actor(obs).detach()
        if self._discrete_actions:
            return {"logits": out}
        return {"mean": out, "log_std": self.log_std.detach().clone()}

    def _fisher_vector_product(self, obs: th.Tensor, old: dict[str, th.Tensor], v: th.Tensor) -> th.Tensor:
        kl = self._mean_kl(obs, old)
        flat_kl_grad = self._flat_grad(kl, create_graph=True)
        gv = (flat_kl_grad * v).sum()
        hv = self._flat_grad(gv)
        return hv + self.cg_damping * v

    @staticmethod
    def _conjugate_gradient(fvp: Callable[[th.Tensor], th.Tensor], b: th.Tensor, iters: int, tol: float = 1e-10) -> th.Tensor:
        x = th.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = th.dot(r, r)
        for _ in range(iters):
            Hp = fvp(p)
            alpha = rdotr / (th.dot(p, Hp) + 1e-8)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = th.dot(r, r)
            if new_rdotr < tol:
                break
            p = r + (new_rdotr / (rdotr + 1e-8)) * p
            rdotr = new_rdotr
        return x

    def _determine_case(self, cost_grad_norm: float, c: float, q: float, r: float, s: float) -> tuple[int, float, float]:
        if cost_grad_norm <= 1e-8 and c < 0:
            return 4, 0.0, 0.0
        A = q - (r ** 2) / (s + 1e-8)
        B = 2.0 * self.target_kl - (c ** 2) / (s + 1e-8)
        if c < 0 and B < 0:
            return 3, A, B          # entire trust region feasible
        if c < 0 and B >= 0:
            return 2, A, B          # partly feasible
        if c >= 0 and B >= 0:
            return 1, A, B          # mostly infeasible (still solvable)
        return 0, A, B              # infeasible -> recovery

    # --- surrogate losses ------------------------------------------------
    def _surrogate_losses(self, obs, actions, old_logp, adv_r, adv_c) -> tuple[th.Tensor, th.Tensor]:
        dist = self._distribution(obs)
        logp = self._action_log_prob(dist, actions)
        ratio = th.exp(logp - old_logp)
        entropy = dist.entropy().mean()
        reward_loss = -(ratio * adv_r).mean() - self.ent_coef * entropy   # minimise -> maximise reward
        cost_loss = (ratio * adv_c).mean()                                # minimise -> reduce cost
        return reward_loss, cost_loss

    # --- training --------------------------------------------------------
    def train(self) -> None:
        adv, ret = _gae(self._rew_buf, self._val_buf, self._done_buf, self._last_value,
                        self.gamma, self.gae_lambda)
        cadv, cret = _gae(self._cost_buf, self._cval_buf, self._done_buf, self._last_cost_value,
                          self.cost_gamma, self.cost_gae_lambda)
        if self.normalize_reward_advantages and len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if self.normalize_cost_advantages:
            cadv = cadv - cadv.mean()

        obs = th.as_tensor(self._obs_buf, dtype=th.float32, device=self.device)
        if self._discrete_actions:
            actions = th.as_tensor(self._act_buf.reshape(-1), dtype=th.long, device=self.device)
        else:
            actions = th.as_tensor(self._act_buf, dtype=th.float32, device=self.device)
        old_logp = th.as_tensor(self._logp_buf, device=self.device)
        adv_r = th.as_tensor(adv, device=self.device)
        adv_c = th.as_tensor(cadv, device=self.device)
        ret_t = th.as_tensor(ret, device=self.device)
        cret_t = th.as_tensor(cret, device=self.device)

        # Surrogate gradients at the current (old) policy.
        reward_loss_before, cost_loss_before = self._surrogate_losses(obs, actions, old_logp, adv_r, adv_c)
        g = self._flat_grad(reward_loss_before, retain_graph=True).detach()   # ascent dir is -g
        b = self._flat_grad(cost_loss_before).detach()

        old = self._old_outputs(obs[::self.fvp_sample_freq])
        obs_fvp = obs[::self.fvp_sample_freq]

        def fvp(v: th.Tensor) -> th.Tensor:
            return self._fisher_vector_product(obs_fvp, old, v)

        x = self._conjugate_gradient(fvp, -g, self.cg_iters)     # ~ H^{-1} g (natural gradient)
        xHx = float(th.dot(x, fvp(x)).item())
        xHx = max(xHx, 1e-8)
        alpha = float(np.sqrt(2.0 * self.target_kl / xHx))

        p = self._conjugate_gradient(fvp, b, self.cg_iters)      # ~ H^{-1} b
        c = float(self._mean_ep_cost - self.cost_limit)
        q = xHx
        r = float(th.dot(-g, p).item())
        s = float(th.dot(b, p).item())

        optim_case, A, B = self._determine_case(float(th.linalg.norm(b).item()), c, q, r, s)

        if optim_case in (3, 4):
            step_dir = alpha * x
            lambda_star, nu_star = 1.0 / (alpha + 1e-8), 0.0
        elif optim_case in (1, 2):
            A_safe = max(A, 0.0)
            B_safe = max(B, 1e-8)
            lambda_a = float(np.sqrt(A_safe / B_safe))
            lambda_b = float(np.sqrt(q / (2.0 * self.target_kl + 1e-8)))
            eps_c = c + 1e-8
            if c < 0:
                lambda_a_star = float(np.clip(lambda_a, 0.0, r / eps_c))
                lambda_b_star = max(lambda_b, r / eps_c)
            else:
                lambda_a_star = max(lambda_a, r / eps_c)
                lambda_b_star = float(np.clip(lambda_b, 0.0, r / eps_c))
            f_a = -0.5 * (A_safe / (lambda_a_star + 1e-8) + B_safe * lambda_a_star) - r * c / (s + 1e-8)
            f_b = -0.5 * (q / (lambda_b_star + 1e-8) + 2.0 * self.target_kl * lambda_b_star)
            lambda_star = lambda_a_star if f_a >= f_b else lambda_b_star
            nu_star = max(0.0, lambda_star * c - r) / (s + 1e-8)
            step_dir = (1.0 / (lambda_star + 1e-8)) * (x - nu_star * p)
        else:
            # Infeasible -> pure recovery step that reduces cost.
            lambda_star, nu_star = 0.0, float(np.sqrt(2.0 * self.target_kl / (s + 1e-8)))
            step_dir = -nu_star * p

        # Backtracking line search. Capture the full-batch old policy (at the
        # pre-step parameters) for the KL check before we start modifying params.
        old_flat = self._flat_params()
        old_full = self._old_outputs(obs)
        accepted = False
        final_kl = 0.0
        for i in range(self.line_search_steps):
            frac = self.line_search_decay ** i
            self._set_flat_params(old_flat + frac * step_dir)
            with th.no_grad():
                cand_reward_loss, cand_cost_loss = self._surrogate_losses(obs, actions, old_logp, adv_r, adv_c)
                kl = float(self._mean_kl(obs, old_full).item())
            reward_improve = float(reward_loss_before.item() - cand_reward_loss.item())
            cost_diff = float(cand_cost_loss.item() - cost_loss_before.item())
            final_kl = kl
            reward_ok = reward_improve > 0 if optim_case > 1 else True
            cost_ok = cost_diff <= max(-c, 0.0)
            kl_ok = kl < self.target_kl
            if reward_ok and cost_ok and kl_ok:
                accepted = True
                break
        if not accepted:
            self._set_flat_params(old_flat)

        # Fit the value functions by regression.
        vf_loss_val = 0.0
        for _ in range(self.n_critic_updates):
            value = self.reward_critic(obs).flatten()
            cost_value = self.cost_critic(obs).flatten()
            vf_loss = ((ret_t - value) ** 2).mean() + ((cret_t - cost_value) ** 2).mean()
            self.critic_optimizer.zero_grad()
            vf_loss.backward()
            self.critic_optimizer.step()
            vf_loss_val = float(vf_loss.item())

        self.last_stats = {
            "reward_policy_loss": float(reward_loss_before.item()),
            "cost_policy_loss": float(cost_loss_before.item()),
            "value_loss": vf_loss_val,
            "kl": final_kl,
            "accepted": float(accepted),
            "optim_case": float(optim_case),
            "lambda_star": float(lambda_star),
            "nu_star": float(nu_star),
            "constraint_value_c": c,
            "mean_episode_cost": float(self._mean_ep_cost),
        }

    def learn(self, total_timesteps: int, callback: Callable[["CPO"], bool] | None = None) -> "CPO":
        while self.num_timesteps < total_timesteps:
            ep_costs = self.collect_rollout()
            self._mean_ep_cost = float(np.mean(ep_costs)) if ep_costs else 0.0
            self.train()
            self.last_stats.update(self._training_violation_stats())
            if callback is not None and not callback(self):
                break
        return self
