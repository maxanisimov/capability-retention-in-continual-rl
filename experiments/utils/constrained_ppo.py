"""Verified-margin constrained PPO baselines for tabular shield safety pipelines.

Shared by the frozenlake, frozenlake_slippery, and lavacrossing safety pipelines
(previously triplicated, byte-for-byte, under each pipeline's ``core/`` package).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_ROOT = _REPO_ROOT / "core"
if str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))

from src.IntervalTensor import IntervalTensor
from src.verification.verify import bound_multi_label_accuracy_margin

from experiments.utils.ppo_utils import (
    PPOConfig,
    _early_stop_thresholds_satisfied,
    _is_early_stop_enabled,
    _warn_if_deprecated_early_stop_settings,
    evaluate_with_success,
    make_actor_critic,
    set_seed,
)

RashomonPayload = dict[str, torch.Tensor]


def validate_rashomon_payload(payload: RashomonPayload) -> None:
    """Structural validation shared across pipelines (column widths are pipeline-specific
    and are expected to have already been checked by the pipeline's own dataset builder)."""
    if set(payload.keys()) != {"state", "actions"}:
        raise ValueError(
            f"Expected Rashomon payload keys {{'state', 'actions'}}, got {sorted(payload.keys())}."
        )
    state = payload["state"]
    actions = payload["actions"]
    if not isinstance(state, torch.Tensor) or not isinstance(actions, torch.Tensor):
        raise TypeError("Rashomon payload values must be torch tensors.")
    if state.dtype != torch.float32 or actions.dtype != torch.float32:
        raise TypeError("Rashomon payload tensors must be float32.")
    if state.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"Expected 2D state/actions tensors, got shapes {tuple(state.shape)} and {tuple(actions.shape)}.",
        )
    if state.shape[0] != actions.shape[0]:
        raise ValueError(
            "Rashomon state/actions tensors must have the same first dimension."
        )
    if not torch.all((actions == 0.0) | (actions == 1.0)):
        raise ValueError("Rashomon action tensor must be multi-hot with 0/1 entries.")


def allowed_action_accuracy(
    actor: torch.nn.Module,
    payload: RashomonPayload,
    *,
    device: str | torch.device = "cpu",
) -> float:
    validate_rashomon_payload(payload)
    states = payload["state"].to(device)
    actions = payload["actions"].to(device)
    actor.eval()
    with torch.no_grad():
        preds = actor(states).argmax(dim=1)
    correct = actions[torch.arange(actions.shape[0], device=actions.device), preds] > 0
    return float(correct.float().mean().item())


@dataclass
class VerifiedMarginConstraint:
    """Source-task safety constraint evaluated through verified margin."""

    states: torch.Tensor
    action_masks: torch.Tensor
    temperature: int
    device: torch.device

    @classmethod
    def from_payload(
        cls,
        payload: RashomonPayload,
        *,
        temperature: int,
        device: str | torch.device,
    ) -> "VerifiedMarginConstraint":
        validate_rashomon_payload(payload)
        device_t = torch.device(device)
        return cls(
            states=payload["state"].to(device_t),
            action_masks=payload["actions"].to(device_t),
            temperature=int(temperature),
            device=device_t,
        )

    def margin_tensor(self, actor: torch.nn.Module) -> torch.Tensor:
        logits = actor(self.states)
        interval_logits = IntervalTensor(logits, logits)
        temperature = int(self.temperature)
        # T=0 (uniform softmax, logits * 0) is a valid calibration rung; tau=inf
        # reproduces it since logits / inf -> 0 -> uniform softmax.
        tau = (1.0 / temperature) if temperature > 0 else float("inf")
        return bound_multi_label_accuracy_margin(
            interval_logits,
            self.action_masks,
            tau=tau,
            lower=True,
            aggregation="min",
        )

    def margin(self, actor: torch.nn.Module) -> float:
        actor.eval()
        with torch.no_grad():
            return float(self.margin_tensor(actor).item())

    def hard_accuracy(self, actor: torch.nn.Module) -> float:
        return allowed_action_accuracy(
            actor,
            {
                "state": self.states.detach().cpu(),
                "actions": self.action_masks.detach().cpu(),
            },
            device=self.device,
        )


@dataclass
class LineSearchDecision:
    accepted: bool
    alpha: float
    margin: float
    backtracks: int


@dataclass
class ConstrainedPPOStats:
    constraint_temperature: int
    calibration_margin: float
    final_margin: float = 0.0
    final_hard_accuracy: float = 0.0
    accepted_updates: int = 0
    direct_accepts: int = 0
    backtracked_accepts: int = 0
    rejected_updates: int = 0
    line_search_attempts: int = 0
    accepted_alphas: list[float] = field(default_factory=list)
    lagrangian_lambda_final: float | None = None
    lagrangian_lambda_updates: int = 0
    lagrangian_mean_violation: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "constraint_temperature": int(self.constraint_temperature),
            "constraint_calibration_margin": float(self.calibration_margin),
            "constraint_final_margin": float(self.final_margin),
            "constraint_final_hard_accuracy": float(self.final_hard_accuracy),
            "constraint_satisfied": bool(self.final_margin > 0.0),
            "safe_line_search_accepted_updates": int(self.accepted_updates),
            "safe_line_search_direct_accepts": int(self.direct_accepts),
            "safe_line_search_backtracked_accepts": int(self.backtracked_accepts),
            "safe_line_search_rejected_updates": int(self.rejected_updates),
            "safe_line_search_attempts": int(self.line_search_attempts),
        }
        if self.accepted_alphas:
            payload.update(
                {
                    "safe_line_search_alpha_min": float(min(self.accepted_alphas)),
                    "safe_line_search_alpha_mean": float(
                        sum(self.accepted_alphas) / len(self.accepted_alphas)
                    ),
                    "safe_line_search_alpha_max": float(max(self.accepted_alphas)),
                },
            )
        if self.lagrangian_lambda_final is not None:
            payload.update(
                {
                    "lagrangian_lambda_final": float(self.lagrangian_lambda_final),
                    "lagrangian_lambda_updates": int(self.lagrangian_lambda_updates),
                },
            )
        if self.lagrangian_mean_violation is not None:
            payload["lagrangian_mean_violation"] = float(self.lagrangian_mean_violation)
        return payload


def calibrate_margin_temperature(
    actor: torch.nn.Module,
    payload: RashomonPayload,
    *,
    inverse_temp_start: int,
    inverse_temp_max: int,
    device: str | torch.device,
) -> tuple[int, float]:
    """Select the first fixed temperature where source safety margin is positive."""
    if inverse_temp_start > inverse_temp_max:
        raise ValueError(
            f"inverse_temp_start must be <= inverse_temp_max, got "
            f"{inverse_temp_start} > {inverse_temp_max}.",
        )
    validate_rashomon_payload(payload)
    actor.to(device)
    actor.eval()
    best_margin = float("-inf")
    with torch.no_grad():
        for temperature in range(int(inverse_temp_start), int(inverse_temp_max) + 1):
            constraint = VerifiedMarginConstraint.from_payload(
                payload,
                temperature=temperature,
                device=device,
            )
            margin = constraint.margin(actor)
            best_margin = margin
            if margin > 0.0:
                return int(temperature), float(margin)
    raise ValueError(
        "Could not calibrate a positive verified multi-label safety margin for "
        f"source policy: last_margin={best_margin:.6f}, "
        f"inverse_temp_range=[{inverse_temp_start}, {inverse_temp_max}].",
    )


def _clone_params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in module.parameters()]


def _load_params(module: torch.nn.Module, values: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, value in zip(module.parameters(), values):
            param.copy_(value.to(device=param.device, dtype=param.dtype))


def _load_interpolated_params(
    module: torch.nn.Module,
    old_values: list[torch.Tensor],
    candidate_values: list[torch.Tensor],
    *,
    alpha: float,
) -> None:
    with torch.no_grad():
        for param, old, candidate in zip(
            module.parameters(), old_values, candidate_values
        ):
            old_t = old.to(device=param.device, dtype=param.dtype)
            candidate_t = candidate.to(device=param.device, dtype=param.dtype)
            param.copy_(old_t + float(alpha) * (candidate_t - old_t))


def apply_safe_line_search(
    actor: torch.nn.Module,
    *,
    old_actor_params: list[torch.Tensor],
    candidate_actor_params: list[torch.Tensor],
    constraint: VerifiedMarginConstraint,
    max_backtracks: int,
    backtrack_coef: float,
) -> LineSearchDecision:
    """Accept, backtrack, or reject a candidate actor update under verified margin."""
    candidate_margin = constraint.margin(actor)
    if candidate_margin > 0.0:
        return LineSearchDecision(
            accepted=True,
            alpha=1.0,
            margin=float(candidate_margin),
            backtracks=0,
        )

    alpha = 1.0
    for backtrack_idx in range(1, int(max_backtracks) + 1):
        alpha *= float(backtrack_coef)
        _load_interpolated_params(
            actor,
            old_actor_params,
            candidate_actor_params,
            alpha=alpha,
        )
        margin = constraint.margin(actor)
        if margin > 0.0:
            return LineSearchDecision(
                accepted=True,
                alpha=float(alpha),
                margin=float(margin),
                backtracks=int(backtrack_idx),
            )

    _load_params(actor, old_actor_params)
    return LineSearchDecision(
        accepted=False,
        alpha=0.0,
        margin=float(constraint.margin(actor)),
        backtracks=int(max_backtracks),
    )


def _constraint_from_calibration(
    actor: torch.nn.Module,
    source_safety_payload: RashomonPayload,
    *,
    inverse_temp_start: int,
    inverse_temp_max: int,
    device: torch.device,
) -> tuple[VerifiedMarginConstraint, ConstrainedPPOStats]:
    temperature, calibration_margin = calibrate_margin_temperature(
        actor,
        source_safety_payload,
        inverse_temp_start=inverse_temp_start,
        inverse_temp_max=inverse_temp_max,
        device=device,
    )
    constraint = VerifiedMarginConstraint.from_payload(
        source_safety_payload,
        temperature=temperature,
        device=device,
    )
    stats = ConstrainedPPOStats(
        constraint_temperature=temperature,
        calibration_margin=calibration_margin,
    )
    return constraint, stats


def _update_final_constraint_stats(
    stats: ConstrainedPPOStats,
    *,
    actor: torch.nn.Module,
    constraint: VerifiedMarginConstraint,
) -> None:
    stats.final_margin = constraint.margin(actor)
    stats.final_hard_accuracy = constraint.hard_accuracy(actor)


def safe_line_search_ppo_train(
    env: gym.Env,
    cfg: PPOConfig,
    *,
    source_safety_payload: RashomonPayload,
    actor_warm_start: torch.nn.Sequential,
    critic_warm_start: torch.nn.Sequential,
    inverse_temp_start: int,
    inverse_temp_max: int,
    max_backtracks: int = 10,
    backtrack_coef: float = 0.5,
    early_stop_eval_env: gym.Env | None = None,
    return_training_data: bool = False,
) -> tuple[
    torch.nn.Sequential,
    torch.nn.Sequential,
    dict[str, np.ndarray] | None,
    ConstrainedPPOStats,
]:
    return _constrained_ppo_train(
        env,
        cfg,
        source_safety_payload=source_safety_payload,
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start,
        inverse_temp_start=inverse_temp_start,
        inverse_temp_max=inverse_temp_max,
        method="safe_line_search",
        max_backtracks=max_backtracks,
        backtrack_coef=backtrack_coef,
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=return_training_data,
    )


def lagrangian_ppo_train(
    env: gym.Env,
    cfg: PPOConfig,
    *,
    source_safety_payload: RashomonPayload,
    actor_warm_start: torch.nn.Sequential,
    critic_warm_start: torch.nn.Sequential,
    inverse_temp_start: int,
    inverse_temp_max: int,
    lambda_init: float = 1.0,
    lambda_lr: float = 0.05,
    lambda_max: float = 100.0,
    early_stop_eval_env: gym.Env | None = None,
    return_training_data: bool = False,
) -> tuple[
    torch.nn.Sequential,
    torch.nn.Sequential,
    dict[str, np.ndarray] | None,
    ConstrainedPPOStats,
]:
    return _constrained_ppo_train(
        env,
        cfg,
        source_safety_payload=source_safety_payload,
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start,
        inverse_temp_start=inverse_temp_start,
        inverse_temp_max=inverse_temp_max,
        method="lagrangian",
        lambda_init=lambda_init,
        lambda_lr=lambda_lr,
        lambda_max=lambda_max,
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=return_training_data,
    )


def _constrained_ppo_train(
    env: gym.Env,
    cfg: PPOConfig,
    *,
    source_safety_payload: RashomonPayload,
    actor_warm_start: torch.nn.Sequential,
    critic_warm_start: torch.nn.Sequential,
    inverse_temp_start: int,
    inverse_temp_max: int,
    method: str,
    max_backtracks: int = 10,
    backtrack_coef: float = 0.5,
    lambda_init: float = 1.0,
    lambda_lr: float = 0.05,
    lambda_max: float = 100.0,
    early_stop_eval_env: gym.Env | None = None,
    return_training_data: bool = False,
) -> tuple[
    torch.nn.Sequential,
    torch.nn.Sequential,
    dict[str, np.ndarray] | None,
    ConstrainedPPOStats,
]:
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError(
            "Verified-margin constrained PPO currently supports discrete action spaces only."
        )
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(
            "Verified-margin constrained PPO expects a Box observation space."
        )
    if method not in {"safe_line_search", "lagrangian"}:
        raise ValueError(f"Unsupported constrained PPO method '{method}'.")
    if max_backtracks < 0:
        raise ValueError(f"max_backtracks must be >= 0, got {max_backtracks}.")
    if not (0.0 < backtrack_coef < 1.0):
        raise ValueError(f"backtrack_coef must be in (0, 1), got {backtrack_coef}.")
    if lambda_init < 0.0 or lambda_lr < 0.0 or lambda_max < 0.0:
        raise ValueError("Lagrangian lambda settings must be non-negative.")

    set_seed(env, cfg.seed)
    _warn_if_deprecated_early_stop_settings(cfg)
    early_stop_enabled = _is_early_stop_enabled(cfg)

    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    actor, critic, log_std = make_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start,
        continuous_actions=False,
    )
    if log_std is not None:
        raise RuntimeError(
            "Unexpected continuous-action log_std for discrete constrained PPO."
        )

    actor_params = list(actor.parameters())
    critic_params = list(critic.parameters())
    device = torch.device(cfg.device)
    actor.to(device)
    critic.to(device)
    constraint, stats = _constraint_from_calibration(
        actor,
        source_safety_payload,
        inverse_temp_start=inverse_temp_start,
        inverse_temp_max=inverse_temp_max,
        device=device,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": actor_params, "lr": cfg.lr},
            {"params": critic_params, "lr": cfg.lr},
        ],
    )
    lagrangian_lambda = float(lambda_init)
    lagrangian_violations: list[float] = []

    eval_env = early_stop_eval_env if early_stop_eval_env is not None else env
    obs, _ = env.reset(seed=cfg.seed)
    global_step = 0
    ppo_update_count = 0
    start_time = time.time()
    stop_training_early = False

    training_data: dict[str, list[Any]] | None = None
    if return_training_data:
        training_data = {
            "states": [],
            "actions": [],
            "terminated": [],
            "truncated": [],
            "safe": [],
        }

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
            stop_training_early = True

    while global_step < cfg.total_timesteps and not stop_training_early:
        obs_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps,), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)

        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs
            obs_t_single = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                value = critic(obs_t_single).squeeze(-1)
                logits = actor(obs_t_single)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                act = int(action.item())

            next_obs, reward, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)

            if training_data is not None:
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
        obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)

        old_actor_params = _clone_params(actor)
        batch_size = T
        idxs = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                actor.train()
                critic.train()
                logits = actor(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value = critic(mb_obs).squeeze(-1)
                value_loss = F.mse_loss(value, mb_ret)
                loss = pg_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                if method == "lagrangian":
                    margin = constraint.margin_tensor(actor)
                    violation = torch.relu(-margin)
                    loss = loss + float(lagrangian_lambda) * violation

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor_params + critic_params, cfg.max_grad_norm
                )
                optimizer.step()

        ppo_update_count += 1

        if method == "safe_line_search":
            candidate_actor_params = _clone_params(actor)
            decision = apply_safe_line_search(
                actor,
                old_actor_params=old_actor_params,
                candidate_actor_params=candidate_actor_params,
                constraint=constraint,
                max_backtracks=max_backtracks,
                backtrack_coef=backtrack_coef,
            )
            stats.line_search_attempts += int(decision.backtracks)
            if decision.accepted:
                stats.accepted_updates += 1
                stats.accepted_alphas.append(float(decision.alpha))
                if decision.alpha == 1.0:
                    stats.direct_accepts += 1
                else:
                    stats.backtracked_accepts += 1
            else:
                stats.rejected_updates += 1
        else:
            current_margin = constraint.margin(actor)
            violation = max(0.0, -float(current_margin))
            lagrangian_violations.append(float(violation))
            lagrangian_lambda = min(
                float(lambda_max),
                max(0.0, lagrangian_lambda + float(lambda_lr) * violation),
            )
            stats.lagrangian_lambda_updates += 1

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
            log_msg = f"Steps={global_step} | meanR={mean_r:.1f} +/- {std_r:.1f} | elapsed={elapsed:.1f}s"
            log_msg += (
                f" | failure_rate={failure_rate:.2f} | success_rate={success_rate:.2f}"
            )
            log_msg += (
                f" | margin={constraint.margin(actor):.6f} | T={constraint.temperature}"
            )
            if method == "safe_line_search":
                log_msg += f" | accepted={stats.accepted_updates} | rejected={stats.rejected_updates}"
            else:
                log_msg += f" | lambda={lagrangian_lambda:.6f}"
            print(log_msg)

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
                        f"success_rate={success_rate:.2f} (threshold={cfg.early_stop_success_rate_threshold})",
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
        final_msg = f"Final evaluation over {cfg.eval_episodes} episodes: mean_reward={mean_r:.2f} +/- {std_r:.2f}"
        final_msg += (
            f" | failure_rate={failure_rate:.2f} | success_rate={success_rate:.2f}"
        )
        final_msg += (
            f" | margin={constraint.margin(actor):.6f} | T={constraint.temperature}"
        )
        print(final_msg)

    if method == "lagrangian":
        stats.lagrangian_lambda_final = float(lagrangian_lambda)
        stats.lagrangian_mean_violation = (
            float(sum(lagrangian_violations) / len(lagrangian_violations))
            if lagrangian_violations
            else 0.0
        )
    _update_final_constraint_stats(stats, actor=actor, constraint=constraint)

    training_arrays: dict[str, np.ndarray] | None = None
    if training_data is not None:
        training_arrays = {
            key: np.asarray(value) for key, value in training_data.items()
        }
    return actor, critic, training_arrays, stats
