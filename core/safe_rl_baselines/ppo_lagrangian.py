"""PPO-Lagrangian: a PyTorch safe-RL baseline (constrained MDP).

A self-contained PyTorch port of MASA's JAX ``PPOLag`` (``masa.algorithms.on_policy.
ppo_lag``). It augments PPO with a **cost critic** and a **Lagrange multiplier** to keep
the discounted cumulative *cost* below a limit:

    maximise  E[reward return]   s.t.   E[cost return] <= cost_limit

Mechanics (matching the reference):
- Separate GAE for reward (``gamma``/``gae_lambda``) and cost
  (``cost_gamma``/``cost_gae_lambda``); two value heads (reward + cost critic).
- Combined advantage ``A = (A_reward - lambda * A_cost) / (1 + lambda)``, with reward
  advantages standardised and cost advantages mean-centred.
- PPO clipped surrogate on ``A``; value loss = ``vf_coef * (MSE(returns, V) +
  MSE(cost_returns, V_cost))``; entropy bonus.
- Naive Lagrange dual update once per rollout:
  ``lambda <- max(0, lambda + lambda_lr * (mean_episode_cost - cost_limit))``.

The per-step cost comes from a user ``cost_fn(obs, action, reward, next_obs, terminated,
truncated, info)`` or, if none is given, from ``info["cost"]`` (safety-gymnasium style).
Supports discrete and box observation/action spaces; a single Gymnasium environment.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces

CostFn = Callable[..., float]


def _mlp(in_dim: int, out_dim: int, net_arch: Sequence[int], activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for size in net_arch:
        layers += [nn.Linear(last, size), activation()]
        last = size
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation. Returns (advantages, returns)."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(n)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


class PPOLagrangian:
    """PPO with a cost critic and a Lagrange multiplier (CMDP baseline)."""

    def __init__(
        self,
        env: gym.Env,
        *,
        net_arch: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # Lagrangian / cost
        cost_limit: float = 25.0,
        cost_gamma: float = 0.99,
        cost_gae_lambda: float = 0.95,
        lagrangian_multiplier_init: float = 0.0,
        lambda_lr: float = 0.01,
        lagrangian_upper_bound: float | None = None,
        normalize_reward_advantages: bool = True,
        normalize_cost_advantages: bool = True,
        cost_fn: CostFn | None = None,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.device = th.device("cuda" if (device == "auto" and th.cuda.is_available()) else
                                (device if device != "auto" else "cpu"))

        self.n_steps = int(n_steps)
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.cost_limit = cost_limit
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.lambda_lr = lambda_lr
        self.lagrangian_upper_bound = lagrangian_upper_bound
        self.lagrangian_multiplier = float(max(lagrangian_multiplier_init, 0.0))
        self.normalize_reward_advantages = normalize_reward_advantages
        self.normalize_cost_advantages = normalize_cost_advantages
        self.cost_fn = cost_fn

        self._rng = np.random.default_rng(seed)
        if seed is not None:
            th.manual_seed(seed)

        # Observation/action dimensions.
        self._obs_dim = self._space_flat_dim(self.observation_space)
        self._discrete_actions = isinstance(self.action_space, spaces.Discrete)
        act_out = int(self.action_space.n) if self._discrete_actions else int(np.prod(self.action_space.shape))

        # Networks: actor + reward critic + cost critic (separate trunks).
        self.actor = _mlp(self._obs_dim, act_out, net_arch, activation).to(self.device)
        self.reward_critic = _mlp(self._obs_dim, 1, net_arch, activation).to(self.device)
        self.cost_critic = _mlp(self._obs_dim, 1, net_arch, activation).to(self.device)
        params = (
            list(self.actor.parameters())
            + list(self.reward_critic.parameters())
            + list(self.cost_critic.parameters())
        )
        if not self._discrete_actions:
            self.log_std = nn.Parameter(th.zeros(act_out, device=self.device))
            params.append(self.log_std)
        self.optimizer = th.optim.Adam(params, lr=learning_rate)

        self.num_timesteps = 0
        self.last_stats: dict[str, float] = {}

    # --- spaces / preprocessing ------------------------------------------
    @staticmethod
    def _space_flat_dim(space: spaces.Space) -> int:
        if isinstance(space, spaces.Discrete):
            return int(space.n)
        return int(np.prod(space.shape))

    def _preprocess(self, obs: np.ndarray) -> th.Tensor:
        """Map a (batch of) observation(s) to a float tensor (one-hot for Discrete)."""
        arr = np.asarray(obs)
        if isinstance(self.observation_space, spaces.Discrete):
            idx = arr.reshape(-1).astype(np.int64)
            onehot = np.zeros((idx.shape[0], int(self.observation_space.n)), dtype=np.float32)
            onehot[np.arange(idx.shape[0]), idx] = 1.0
            return th.as_tensor(onehot, device=self.device)
        return th.as_tensor(arr.reshape(arr.shape[0] if arr.ndim > 1 else 1, -1), dtype=th.float32, device=self.device)

    # --- policy ----------------------------------------------------------
    def _distribution(self, obs_t: th.Tensor) -> th.distributions.Distribution:
        out = self.actor(obs_t)
        if self._discrete_actions:
            return th.distributions.Categorical(logits=out)
        return th.distributions.Normal(out, self.log_std.exp())

    def _action_log_prob(self, dist: th.distributions.Distribution, actions: th.Tensor) -> th.Tensor:
        if self._discrete_actions:
            return dist.log_prob(actions)
        return dist.log_prob(actions).sum(-1)

    @th.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, None]:
        obs_t = self._preprocess(obs)
        dist = self._distribution(obs_t)
        if deterministic:
            action = dist.probs.argmax(-1) if self._discrete_actions else dist.mean
        else:
            action = dist.sample()
        action_np = action.cpu().numpy()
        if np.asarray(obs).ndim == (1 if isinstance(self.observation_space, spaces.Discrete) else
                                    len(self.observation_space.shape)):
            action_np = action_np[0]
        return action_np, None

    # --- rollout ---------------------------------------------------------
    def _step_cost(self, obs, action, reward, next_obs, terminated, truncated, info) -> float:
        if self.cost_fn is not None:
            return float(self.cost_fn(obs, action, reward, next_obs, terminated, truncated, info))
        return float(info.get("cost", 0.0))

    def collect_rollout(self) -> list[float]:
        """Collect ``n_steps`` transitions; fill buffers. Returns completed-episode costs."""
        n = self.n_steps
        self._obs_buf = np.zeros((n,) + self._obs_shape(), dtype=np.float32)
        self._act_buf = np.zeros((n,) + ((1,) if self._discrete_actions else self.action_space.shape), dtype=np.float32)
        self._logp_buf = np.zeros(n, dtype=np.float32)
        self._rew_buf = np.zeros(n, dtype=np.float32)
        self._cost_buf = np.zeros(n, dtype=np.float32)
        self._val_buf = np.zeros(n, dtype=np.float32)
        self._cval_buf = np.zeros(n, dtype=np.float32)
        self._done_buf = np.zeros(n, dtype=np.float32)

        ep_costs: list[float] = []
        if not hasattr(self, "_last_obs") or self._last_obs is None:
            self._last_obs, _ = self.env.reset()
            self._ep_cost = 0.0

        for t in range(n):
            obs_t = self._preprocess(self._last_obs)
            with th.no_grad():
                dist = self._distribution(obs_t)
                action = dist.sample()
                log_prob = self._action_log_prob(dist, action)
                value = self.reward_critic(obs_t).flatten()
                cost_value = self.cost_critic(obs_t).flatten()
            action_env = int(action.item()) if self._discrete_actions else action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, info = self.env.step(action_env)
            cost = self._step_cost(self._last_obs, action_env, reward, next_obs, terminated, truncated, info)

            self._obs_buf[t] = self._preprocess(self._last_obs).cpu().numpy()[0]
            self._act_buf[t] = action.cpu().numpy().reshape(self._act_buf.shape[1:])
            self._logp_buf[t] = float(log_prob.item())
            self._rew_buf[t] = float(reward)
            self._cost_buf[t] = cost
            self._val_buf[t] = float(value.item())
            self._cval_buf[t] = float(cost_value.item())
            done = bool(terminated or truncated)
            self._done_buf[t] = float(done)
            self._ep_cost += cost
            self.num_timesteps += 1

            if truncated and not terminated:
                # Bootstrap past truncation for both critics.
                with th.no_grad():
                    nxt = self._preprocess(next_obs)
                    self._rew_buf[t] += self.gamma * float(self.reward_critic(nxt).item())
                    self._cost_buf[t] += self.cost_gamma * float(self.cost_critic(nxt).item())

            if done:
                ep_costs.append(self._ep_cost)
                self._ep_cost = 0.0
                next_obs, _ = self.env.reset()
            self._last_obs = next_obs

        with th.no_grad():
            last_t = self._preprocess(self._last_obs)
            self._last_value = float(self.reward_critic(last_t).item())
            self._last_cost_value = float(self.cost_critic(last_t).item())
        return ep_costs

    def _obs_shape(self) -> tuple[int, ...]:
        return (self._obs_dim,)

    # --- training --------------------------------------------------------
    def _update_lagrange_multiplier(self, mean_ep_cost: float) -> None:
        self.lagrangian_multiplier += self.lambda_lr * (mean_ep_cost - self.cost_limit)
        self.lagrangian_multiplier = max(0.0, self.lagrangian_multiplier)
        if self.lagrangian_upper_bound is not None:
            self.lagrangian_multiplier = min(self.lagrangian_multiplier, self.lagrangian_upper_bound)

    def train(self) -> None:
        # Returns and advantages for reward and cost.
        adv, ret = _gae(self._rew_buf, self._val_buf, self._done_buf, self._last_value,
                        self.gamma, self.gae_lambda)
        cadv, cret = _gae(self._cost_buf, self._cval_buf, self._done_buf, self._last_cost_value,
                          self.cost_gamma, self.cost_gae_lambda)

        obs = th.as_tensor(self._obs_buf, dtype=th.float32, device=self.device)
        if self._discrete_actions:
            actions = th.as_tensor(self._act_buf.reshape(-1), dtype=th.long, device=self.device)
        else:
            actions = th.as_tensor(self._act_buf, dtype=th.float32, device=self.device)
        old_log_prob = th.as_tensor(self._logp_buf, device=self.device)
        ret_t = th.as_tensor(ret, device=self.device)
        cret_t = th.as_tensor(cret, device=self.device)
        adv_t = th.as_tensor(adv, device=self.device)
        cadv_t = th.as_tensor(cadv, device=self.device)

        n = self.n_steps
        lam = self.lagrangian_multiplier
        pg_loss_val = vf_loss_val = 0.0
        for _ in range(self.n_epochs):
            for idx in th.randperm(n, device=self.device).split(self.batch_size):
                b_adv = adv_t[idx]
                b_cadv = cadv_t[idx]
                if self.normalize_reward_advantages and len(idx) > 1:
                    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                if self.normalize_cost_advantages:
                    b_cadv = b_cadv - b_cadv.mean()
                combined_adv = (b_adv - lam * b_cadv) / (1.0 + lam)

                dist = self._distribution(obs[idx])
                log_prob = self._action_log_prob(dist, actions[idx])
                ratio = th.exp(log_prob - old_log_prob[idx])
                p1 = combined_adv * ratio
                p2 = combined_adv * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(p1, p2).mean()
                entropy_loss = -dist.entropy().mean()

                value = self.reward_critic(obs[idx]).flatten()
                cost_value = self.cost_critic(obs[idx]).flatten()
                value_loss = ((ret_t[idx] - value) ** 2).mean() + ((cret_t[idx] - cost_value) ** 2).mean()

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]], self.max_grad_norm
                )
                self.optimizer.step()
                pg_loss_val = float(policy_loss.item())
                vf_loss_val = float(value_loss.item())

        self.last_stats = {
            "policy_loss": pg_loss_val,
            "value_loss": vf_loss_val,
            "lagrangian_multiplier": float(self.lagrangian_multiplier),
            "mean_cost_advantage": float(cadv.mean()),
            "mean_cost_return": float(cret.mean()),
        }

    def learn(self, total_timesteps: int) -> "PPOLagrangian":
        while self.num_timesteps < total_timesteps:
            ep_costs = self.collect_rollout()
            mean_ep_cost = float(np.mean(ep_costs)) if ep_costs else None
            if mean_ep_cost is not None:
                # Update the dual variable before optimisation (train() uses current lambda).
                self._update_lagrange_multiplier(mean_ep_cost)
            self.train()
            if mean_ep_cost is not None:
                self.last_stats["mean_episode_cost"] = mean_ep_cost
        return self
