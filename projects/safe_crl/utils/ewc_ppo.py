"""
Elastic Weight Consolidation (EWC) wrapper for PPO.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks", PNAS 2017.  https://arxiv.org/abs/1612.00796

Typical continual-learning workflow
------------------------------------
1.  Train on task 1 with plain PPO (ppo_utils.ppo_train).
2.  Collect a set of representative observations from task 1.
3.  Call `compute_ewc_state(actor, observations, ...)` to build an EWCState
    that stores the Fisher diagonal and the anchor ("star") parameters.
4.  Train on task 2 with `ewc_ppo_train(..., ewc_states=[state_task1])`.
    The EWC penalty keeps the actor close to its task-1 optimum, weighted
    by parameter importance.
5.  Repeat: build a new EWCState from the task-2-trained actor and pass both
    states to `ewc_ppo_train` for task 3, etc.

EWC penalty
-----------
Given a list of EWCState objects (one per previous task t), the penalty is:

    L_EWC = (ewc_lambda / 2) * Σ_t Σ_i  F^t_i * (θ_i − θ^*_t_i)^2

where F^t_i is the empirical Fisher diagonal for parameter i on task t, and
θ^*_t_i is the parameter value at the end of task t training.

This penalty is added to the standard PPO loss at every gradient step.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppo_utils import (
    PPOConfig,
    _early_stop_thresholds_satisfied,
    _is_early_stop_enabled,
    _warn_if_deprecated_early_stop_settings,
    evaluate_with_success,
    make_actor_critic,
    set_seed,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EWCPPOConfig(PPOConfig):
    """PPO configuration extended with EWC hyperparameters."""
    ewc_lambda: float = 5_000.0
    """Strength of the EWC regularisation penalty."""
    ewc_apply_to_critic: bool = False
    """Whether to also regularise the critic with EWC (rare but supported)."""


# ---------------------------------------------------------------------------
# EWC state (one per previous task)
# ---------------------------------------------------------------------------

@dataclass
class EWCState:
    """
    Stores per-parameter Fisher diagonals and anchor weights for one task.

    Attributes
    ----------
    fisher_actor : dict[str, Tensor]
        Diagonal of the empirical Fisher information matrix for each actor
        parameter, keyed by ``name`` from ``actor.named_parameters()``.
    star_actor : dict[str, Tensor]
        Snapshot of the actor parameters at the end of task training.
    fisher_critic : dict[str, Tensor] | None
        Same as ``fisher_actor`` but for the critic.  ``None`` when the
        critic was not regularised.
    star_critic : dict[str, Tensor] | None
        Snapshot of the critic parameters.  ``None`` when not regularised.
    """
    fisher_actor: dict[str, torch.Tensor]
    star_actor: dict[str, torch.Tensor]
    fisher_critic: dict[str, torch.Tensor] | None = None
    star_critic: dict[str, torch.Tensor] | None = None


# ---------------------------------------------------------------------------
# Fisher computation
# ---------------------------------------------------------------------------

def compute_ewc_state(
    actor: nn.Module,
    observations: np.ndarray,
    *,
    compute_critic: bool = False,
    critic: nn.Module | None = None,
    log_std: nn.Parameter | None = None,
    device: str | torch.device = "cpu",
    fisher_sample_size: int | None = None,
    seed: int = 0,
) -> EWCState:
    """
    Compute an EWCState from the current (just-trained) actor weights.

    The empirical Fisher diagonal is estimated by sampling actions from the
    current policy for each observed state, then computing the expected
    squared gradient of log π(a|s) w.r.t. all actor parameters.

    Parameters
    ----------
    actor : nn.Module
        The trained actor network.
    observations : np.ndarray, shape (N, obs_dim)
        Representative states from the just-completed task.  These are
        typically the states visited during training (from
        ``ppo_train(..., return_training_data=True)``).
    compute_critic : bool
        If True, also compute Fisher information for the critic.
    critic : nn.Module | None
        Required when ``compute_critic=True``.
    log_std : nn.Parameter | None
        Required for continuous-action actors.
    device : str | torch.device
        Device to run computation on.
    fisher_sample_size : int | None
        Maximum number of observations to use.  If None, uses all of them.
        Randomly sub-sampled when fewer than N are requested.
    seed : int
        Seed for sub-sampling reproducibility.

    Returns
    -------
    EWCState
    """
    device = torch.device(device)
    actor = actor.to(device)
    actor.eval()

    rng = np.random.default_rng(seed)

    # Optional sub-sampling
    if fisher_sample_size is not None and fisher_sample_size < len(observations):
        idx = rng.choice(len(observations), size=fisher_sample_size, replace=False)
        observations = observations[idx]

    # Detect action space type from actor output dimension vs log_std
    continuous_actions = log_std is not None

    # ---- Actor Fisher -------------------------------------------------------
    fisher_actor: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in actor.named_parameters()
    }

    n_samples = len(observations)
    for obs in observations:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        actor.zero_grad()

        if continuous_actions:
            mean = actor(obs_t)
            std = torch.exp(log_std)  # type: ignore[arg-type]
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            logits = actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        log_prob.backward()

        with torch.no_grad():
            for name, param in actor.named_parameters():
                if param.grad is not None:
                    fisher_actor[name] += param.grad.detach() ** 2

    # Normalise by sample count and detach
    with torch.no_grad():
        for name in fisher_actor:
            fisher_actor[name] = (fisher_actor[name] / n_samples).detach()

    # Anchor parameters: deep-copy of current weights (detached)
    star_actor: dict[str, torch.Tensor] = {
        name: param.detach().clone()
        for name, param in actor.named_parameters()
    }

    actor.train()

    # ---- Critic Fisher (optional) -------------------------------------------
    fisher_critic: dict[str, torch.Tensor] | None = None
    star_critic: dict[str, torch.Tensor] | None = None

    if compute_critic:
        assert critic is not None, "critic must be provided when compute_critic=True"
        critic = critic.to(device)
        critic.eval()

        fisher_critic = {
            name: torch.zeros_like(param, device=device)
            for name, param in critic.named_parameters()
        }

        for obs in observations:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            critic.zero_grad()
            value = critic(obs_t).squeeze(-1)
            # Use squared value as a proxy for critic importance
            # (true critic Fisher requires a distribution over returns)
            value.backward()
            with torch.no_grad():
                for name, param in critic.named_parameters():
                    if param.grad is not None:
                        fisher_critic[name] += param.grad.detach() ** 2

        with torch.no_grad():
            for name in fisher_critic:
                fisher_critic[name] = (fisher_critic[name] / n_samples).detach()

        star_critic = {
            name: param.detach().clone()
            for name, param in critic.named_parameters()
        }
        critic.train()

    return EWCState(
        fisher_actor=fisher_actor,
        star_actor=star_actor,
        fisher_critic=fisher_critic,
        star_critic=star_critic,
    )


# ---------------------------------------------------------------------------
# EWC penalty
# ---------------------------------------------------------------------------

def ewc_penalty(
    module: nn.Module,
    ewc_states: list[EWCState],
    fisher_key: str = "fisher_actor",
    star_key: str = "star_actor",
) -> torch.Tensor:
    """
    Compute the (un-scaled) EWC regularisation penalty for *module*.

    The caller is responsible for multiplying by ``ewc_lambda / 2``.

    Parameters
    ----------
    module : nn.Module
        Network whose parameters are being regularised (actor or critic).
    ewc_states : list[EWCState]
        Previous-task EWC states to regularise against.
    fisher_key : str
        Attribute of EWCState that contains the Fisher dict for this module.
    star_key : str
        Attribute of EWCState that contains the anchor-param dict.

    Returns
    -------
    Tensor (scalar)
    """
    penalty = torch.tensor(0.0, device=next(module.parameters()).device)
    for state in ewc_states:
        fisher: dict[str, torch.Tensor] = getattr(state, fisher_key)
        star: dict[str, torch.Tensor] = getattr(state, star_key)
        if fisher is None or star is None:
            continue
        for name, param in module.named_parameters():
            f = fisher.get(name)
            s = star.get(name)
            if f is not None and s is not None:
                penalty = penalty + (f * (param - s.to(param.device)) ** 2).sum()
    return penalty


# ---------------------------------------------------------------------------
# EWC-augmented PPO training
# ---------------------------------------------------------------------------

def ewc_ppo_train(
    env: gym.Env,
    cfg: EWCPPOConfig,
    ewc_states: list[EWCState] | None = None,
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
    actor_param_bounds_l: list[torch.Tensor] | None = None,
    actor_param_bounds_u: list[torch.Tensor] | None = None,
    early_stop_eval_env: gym.Env | None = None,
    return_training_data: bool = False,
):
    """
    Train a PPO agent with an Elastic Weight Consolidation penalty.

    This is a drop-in replacement for ``ppo_utils.ppo_train`` that accepts an
    optional list of ``EWCState`` objects (one per previous task).  When
    ``ewc_states`` is ``None`` or empty, training degenerates to plain PPO.

    At every gradient step the total loss is:

        L = L_PPO  +  (ewc_lambda / 2) * Σ_t Σ_i  F^t_i * (θ_i − θ^*_t_i)^2

    Parameters
    ----------
    env : gym.Env
        Training environment.
    cfg : EWCPPOConfig
        Hyperparameters (includes PPOConfig fields plus ``ewc_lambda`` and
        ``ewc_apply_to_critic``).
    ewc_states : list[EWCState] | None
        EWC states from all previous tasks.  Pass ``None`` or ``[]`` for
        standard PPO.
    actor_warm_start : nn.Sequential | None
        Pre-trained actor to start from (e.g. the task-N actor).
    critic_warm_start : nn.Sequential | None
        Pre-trained critic to start from.
    actor_param_bounds_l / actor_param_bounds_u : list[Tensor] | None
        Optional projected-gradient-descent bounds (same as ppo_train).
    early_stop_eval_env : gym.Env | None
        Optional environment for periodic evaluation and early-stopping checks.
        If None, uses the training environment.
    return_training_data : bool
        If True, also return a dict of visited (state, action, …) pairs
        (useful for building the EWCState for the *next* task).

    Returns
    -------
    Without return_training_data
        - Discrete: ``(actor, critic)``
        - Continuous: ``(actor, critic, log_std)``
    With return_training_data
        - Discrete: ``(actor, critic, training_data)``
        - Continuous: ``(actor, critic, log_std, training_data)``
    """
    active_ewc = bool(ewc_states)  # False when None or empty list
    ewc_states = ewc_states or []

    set_seed(env, cfg.seed)
    _warn_if_deprecated_early_stop_settings(cfg)
    early_stop_enabled = _is_early_stop_enabled(cfg)

    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    obs_dim = (
        env.observation_space.shape[0]
        if isinstance(env.observation_space, gym.spaces.Box)
        else env.observation_space.n  # type: ignore[attr-defined]
    )
    n_actions = (
        env.action_space.shape[0] if continuous_actions else env.action_space.n  # type: ignore[attr-defined]
    )

    actor, critic, log_std = make_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,  # type: ignore[arg-type]
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start,
        continuous_actions=continuous_actions,
    )
    device = torch.device(cfg.device)
    actor.to(device)
    critic.to(device)
    if log_std is not None:
        log_std = log_std.to(device)

    # Move EWC state tensors to the same device
    for state in ewc_states:
        state.fisher_actor = {k: v.to(device) for k, v in state.fisher_actor.items()}
        state.star_actor = {k: v.to(device) for k, v in state.star_actor.items()}
        if state.fisher_critic is not None:
            state.fisher_critic = {k: v.to(device) for k, v in state.fisher_critic.items()}
        if state.star_critic is not None:
            state.star_critic = {k: v.to(device) for k, v in state.star_critic.items()}

    # Projected gradient descent setup
    use_pgd = (actor_param_bounds_l is not None and actor_param_bounds_u is not None)
    print(f"Use PGD: {use_pgd} | EWC active: {active_ewc} (lambda={cfg.ewc_lambda if active_ewc else 'N/A'}, {len(ewc_states)} task(s))")
    bounds_l = bounds_u = None
    if use_pgd:
        bounds_l = [b.to(device) for b in actor_param_bounds_l]  # type: ignore[union-attr]
        bounds_u = [b.to(device) for b in actor_param_bounds_u]  # type: ignore[union-attr]
        with torch.no_grad():
            for param, lb, ub in zip(actor.parameters(), bounds_l, bounds_u):
                param.data.clamp_(lb, ub)

    optimizer_params = [
        {"params": actor.parameters(), "lr": cfg.lr},
        {"params": critic.parameters(), "lr": cfg.lr},
    ]
    if log_std is not None:
        optimizer_params.append({"params": [log_std], "lr": cfg.lr})
    optimizer = torch.optim.Adam(optimizer_params)
    eval_env = early_stop_eval_env if early_stop_eval_env is not None else env

    obs, _ = env.reset(seed=cfg.seed)
    global_step = 0
    ppo_update_count = 0
    start_time = time.time()
    pgd_projections = 0
    stop_training_early = False

    if return_training_data:
        training_data: dict = {
            "states": [],
            "actions": [],
            "terminated": [],
            "truncated": [],
            "safe": [],
        }

    # Initial pre-update early-stop check at step=0.
    if early_stop_enabled and cfg.early_stop_min_steps == 0:
        mean_r, std_r, failure_rate, success_rate = evaluate_with_success(
            env=eval_env,
            actor=actor,
            device=device,
            episodes=cfg.eval_episodes,
            deterministic=True,
        )
        if eval_env is env:
            obs, _ = env.reset(seed=cfg.seed)
        print(
            f"Pre-update check | Steps=0 | meanR={mean_r:.1f} +/- {std_r:.1f} | "
            f"failure_rate={failure_rate:.2f} | success_rate={success_rate:.2f}",
        )
        reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
            cfg,
            mean_reward=mean_r,
            failure_rate=failure_rate,
            success_rate=success_rate,
        )
        if reward_ok and failure_ok and success_ok:
            print(
                "  [Early stop pre-update] step=0 | "
                f"meanR={mean_r:.2f} (threshold={cfg.early_stop_reward_threshold}) | "
                f"failure_rate={failure_rate:.2f} (threshold={cfg.early_stop_failure_rate_threshold}) | "
                f"success_rate={success_rate:.2f} (threshold={cfg.early_stop_success_rate_threshold})"
            )
            stop_training_early = True

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    while global_step < cfg.total_timesteps and not stop_training_early:
        # --- Rollout collection -------------------------------------------
        obs_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
        if continuous_actions:
            act_buf = np.zeros((cfg.rollout_steps, n_actions), dtype=np.float32)  # type: ignore[arg-type]
        else:
            act_buf = np.zeros((cfg.rollout_steps,), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)

        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                value = critic(obs_t).squeeze(-1)
                if continuous_actions:
                    mean = actor(obs_t)
                    std = torch.exp(log_std)  # type: ignore[arg-type]
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    logp = dist.log_prob(action).sum(dim=-1)
                    action_np = np.clip(
                        action.cpu().numpy()[0],
                        env.action_space.low,  # type: ignore[union-attr]
                        env.action_space.high,  # type: ignore[union-attr]
                    )
                    act = action_np
                else:
                    logits = actor(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    act = int(action.item())

            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            if return_training_data:
                training_data["states"].append(obs.copy())
                training_data["actions"].append(act)
                training_data["terminated"].append(float(terminated))
                training_data["truncated"].append(float(truncated))
                is_safe = info.get("safe", None)
                if is_safe is None:
                    cost = info.get("cost", 0)
                    is_safe = 1.0 if cost == 0 else 0.0
                training_data["safe"].append(float(is_safe))

            act_buf[t] = act
            logp_buf[t] = float(logp.item())
            rew_buf[t] = float(reward)
            done_buf[t] = float(done)
            val_buf[t] = float(value.item())

            obs = next_obs
            global_step += 1
            if done:
                obs, _ = env.reset()

            if global_step >= cfg.total_timesteps:
                obs_buf = obs_buf[: t + 1]
                act_buf = act_buf[: t + 1]
                logp_buf = logp_buf[: t + 1]
                rew_buf = rew_buf[: t + 1]
                done_buf = done_buf[: t + 1]
                val_buf = val_buf[: t + 1]
                break

        # --- GAE -----------------------------------------------------------
        with torch.no_grad():
            last_val = critic(
                torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ).item()

        T = len(rew_buf)
        adv_buf = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - done_buf[t]
            next_value = last_val if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        adv_t = torch.tensor(adv_buf, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        obs_t_all = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_t_all = (
            torch.tensor(act_buf, dtype=torch.float32, device=device)
            if continuous_actions
            else torch.tensor(act_buf, dtype=torch.int64, device=device)
        )
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)

        # --- PPO updates with EWC penalty ---------------------------------
        batch_size = T
        idxs = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t_all[mb_idx]
                mb_act = act_t_all[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                if continuous_actions:
                    mean = actor(mb_obs)
                    std = torch.exp(log_std)  # type: ignore[arg-type]
                    dist = torch.distributions.Normal(mean, std)
                    new_logp = dist.log_prob(mb_act).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    logits = actor(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()

                v = critic(mb_obs).squeeze(-1)
                v_loss = F.mse_loss(v, mb_ret)

                ppo_loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                # EWC penalty
                ewc_loss = torch.tensor(0.0, device=device)
                if active_ewc:
                    ewc_loss = ewc_loss + ewc_penalty(
                        actor, ewc_states,
                        fisher_key="fisher_actor", star_key="star_actor",
                    )
                    if cfg.ewc_apply_to_critic:
                        ewc_loss = ewc_loss + ewc_penalty(
                            critic, ewc_states,
                            fisher_key="fisher_critic", star_key="star_critic",
                        )

                loss = ppo_loss + (cfg.ewc_lambda / 2.0) * ewc_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                all_params = list(actor.parameters()) + list(critic.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
                optimizer.step()

                # Projected gradient descent: clamp actor params to bounds
                if use_pgd:
                    with torch.no_grad():
                        for param, lb, ub in zip(actor.parameters(), bounds_l, bounds_u):  # type: ignore[arg-type]
                            violations = ((param.data < lb) | (param.data > ub)).sum().item()
                            if violations > 0:
                                pgd_projections += violations
                            param.data.clamp_(lb, ub)
        ppo_update_count += 1

        # --- Periodic evaluation and logging ------------------------------
        if global_step % (10 * cfg.rollout_steps) < cfg.rollout_steps:
            mean_r, std_r, failure_rate, success_rate = evaluate_with_success(
                env=eval_env,
                actor=actor,
                device=device,
                episodes=cfg.eval_episodes,
                deterministic=True,
            )
            if eval_env is env:
                obs, _ = env.reset()
            elapsed = time.time() - start_time
            log_msg = (
                f"Steps={global_step} | meanR={mean_r:.1f} +/- {std_r:.1f}"
                f" | elapsed={elapsed:.1f}s | failure_rate={failure_rate:.2f}"
            )
            log_msg += f" | success_rate={success_rate:.2f}"
            if active_ewc:
                log_msg += f" | EWC lambda={cfg.ewc_lambda}"
            if use_pgd:
                log_msg += f" | PGD projections={pgd_projections}"
            print(log_msg)

            # Early stopping
            if early_stop_enabled and ppo_update_count >= cfg.early_stop_min_steps:
                reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
                    cfg,
                    mean_reward=mean_r,
                    failure_rate=failure_rate,
                    success_rate=success_rate,
                )
                if reward_ok and failure_ok and success_ok:
                    print(
                        f"  [Early stop] updates={ppo_update_count} | step={global_step} | "
                        f"meanR={mean_r:.2f} (threshold={cfg.early_stop_reward_threshold}) | "
                        f"failure_rate={failure_rate:.2f} (threshold={cfg.early_stop_failure_rate_threshold}) | "
                        f"success_rate={success_rate:.2f} (threshold={cfg.early_stop_success_rate_threshold})"
                    )
                    break

    if cfg.eval_episodes is not None and cfg.eval_episodes > 0:
        mean_r, std_r, failure_rate, success_rate = evaluate_with_success(
            env=eval_env,
            actor=actor,
            device=device,
            episodes=cfg.eval_episodes,
            deterministic=True,
        )
        final_msg = (
            f"Final evaluation over {cfg.eval_episodes} episodes: "
            f"mean_reward={mean_r:.2f} +/- {std_r:.2f}"
            f" | failure_rate={failure_rate:.2f}"
            f" | success_rate={success_rate:.2f}"
        )
        if use_pgd:
            final_msg += f" | Total PGD projections: {pgd_projections}"
        print(final_msg)

    env.close()
    if early_stop_eval_env is not None and early_stop_eval_env is not env:
        early_stop_eval_env.close()

    # Return results
    if return_training_data:
        training_data["states"] = np.array(training_data["states"])
        training_data["actions"] = np.array(training_data["actions"])
        training_data["terminated"] = np.array(training_data["terminated"])
        training_data["truncated"] = np.array(training_data["truncated"])
        training_data["safe"] = np.array(training_data["safe"])
        if continuous_actions:
            return actor, critic, log_std, training_data
        else:
            return actor, critic, training_data
    else:
        if continuous_actions:
            return actor, critic, log_std
        else:
            return actor, critic
