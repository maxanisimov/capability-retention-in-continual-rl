"""PPO with a hard source-demonstration distillation penalty."""

from __future__ import annotations

from dataclasses import dataclass
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from projects.safe_crl.utils.ppo_utils import (
    PPOConfig,
    _early_stop_thresholds_satisfied,
    _is_early_stop_enabled,
    _warn_if_deprecated_early_stop_settings,
    evaluate_with_success,
    make_actor_critic,
    set_seed,
)


@dataclass
class DistillationPPOConfig(PPOConfig):
    """PPO configuration extended with source-demonstration distillation."""

    distill_lambda: float = 1.0
    distill_batch_size: int | None = None


def _demo_tensors(
    source_demo_dataset: TensorDataset | tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(source_demo_dataset, TensorDataset):
        if len(source_demo_dataset.tensors) < 2:
            raise ValueError("source_demo_dataset must contain observation and label tensors.")
        obs, labels = source_demo_dataset.tensors[:2]
    else:
        obs, labels = source_demo_dataset
    if int(obs.shape[0]) <= 0:
        raise ValueError("source_demo_dataset is empty.")
    if int(obs.shape[0]) != int(labels.shape[0]):
        raise ValueError(
            "source_demo_dataset observation/label length mismatch: "
            f"{int(obs.shape[0])} != {int(labels.shape[0])}.",
        )
    obs = obs.detach().float().to(device)
    labels = labels.detach().to(device)
    if labels.ndim == 2:
        if int(labels.shape[1]) <= 0:
            raise ValueError("source_demo_dataset has zero action columns.")
        action_labels = torch.argmax(labels.float(), dim=1).long()
    elif labels.ndim == 1:
        action_labels = labels.long()
    else:
        raise ValueError(f"Unsupported source_demo_dataset label shape: {tuple(labels.shape)}.")
    return obs, action_labels


def demonstration_metrics(
    actor: nn.Module,
    source_demo_dataset: TensorDataset | tuple[torch.Tensor, torch.Tensor],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Return empirical retention metrics on the source-demonstration dataset."""
    eval_device = torch.device(device)
    actor_was_training = actor.training
    actor = actor.to(eval_device)
    actor.eval()
    obs, actions = _demo_tensors(source_demo_dataset, device=eval_device)
    with torch.no_grad():
        logits = actor(obs)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        selected_prob = probs.gather(1, actions.view(-1, 1)).squeeze(1)
        ce = F.cross_entropy(logits, actions)
    if actor_was_training:
        actor.train()
    return {
        "source_demo_accuracy": float((pred == actions).float().mean().item()),
        "source_demo_mean_action_probability": float(selected_prob.mean().item()),
        "source_demo_cross_entropy": float(ce.item()),
    }


def _sample_demo_minibatch(
    demo_obs: torch.Tensor,
    demo_actions: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(demo_obs.shape[0])
    if n <= 0:
        raise ValueError("Cannot sample from an empty source demonstration dataset.")
    idx = torch.randint(0, n, (int(batch_size),), device=demo_obs.device)
    return demo_obs[idx], demo_actions[idx]


def distillation_ppo_train(
    env: gym.Env,
    cfg: DistillationPPOConfig,
    *,
    source_demo_dataset: TensorDataset | tuple[torch.Tensor, torch.Tensor],
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
    early_stop_eval_env: gym.Env | None = None,
    return_training_data: bool = False,
):
    """Train discrete-action PPO with a hard CE penalty to source demonstrations."""
    set_seed(env, cfg.seed)
    _warn_if_deprecated_early_stop_settings(cfg)
    early_stop_enabled = _is_early_stop_enabled(cfg)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("distillation_ppo_train only supports discrete action spaces.")
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("distillation_ppo_train expects a Box observation space.")

    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    actor, critic, _ = make_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start,
        continuous_actions=False,
    )
    actor_params = list(actor.parameters())
    critic_params = list(critic.parameters())
    device = torch.device(cfg.device)
    actor.to(device)
    critic.to(device)
    demo_obs, demo_actions = _demo_tensors(source_demo_dataset, device=device)
    if int(demo_actions.max().item()) >= n_actions or int(demo_actions.min().item()) < 0:
        raise ValueError("source_demo_dataset contains action labels outside the env action space.")

    distill_batch_size = int(cfg.distill_batch_size or cfg.minibatch_size)
    if distill_batch_size <= 0:
        raise ValueError(f"distill_batch_size must be > 0, got {distill_batch_size}.")

    optimizer = torch.optim.Adam(
        [
            {"params": actor_params, "lr": cfg.lr},
            {"params": critic_params, "lr": cfg.lr},
        ],
    )
    eval_env = early_stop_eval_env if early_stop_eval_env is not None else env
    obs, _ = env.reset(seed=cfg.seed)
    global_step = 0
    ppo_update_count = 0
    start_time = time.time()
    stop_training_early = False

    if return_training_data:
        training_data: dict[str, list] = {
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
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                value = critic(obs_t).squeeze(-1)
                dist = torch.distributions.Categorical(logits=actor(obs_t))
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
                    is_safe = 1.0 if info.get("cost", 0) == 0 else 0.0
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
            last_val = critic(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()

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
        act_t_all = torch.tensor(act_buf, dtype=torch.int64, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)

        idxs = np.arange(T)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, cfg.minibatch_size):
                mb_idx = idxs[start : start + cfg.minibatch_size]
                mb_obs = obs_t_all[mb_idx]
                mb_act = act_t_all[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                dist = torch.distributions.Categorical(logits=actor(mb_obs))
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_logp - mb_old_logp)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()
                v_loss = F.mse_loss(critic(mb_obs).squeeze(-1), mb_ret)
                ppo_loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                demo_mb_obs, demo_mb_actions = _sample_demo_minibatch(
                    demo_obs,
                    demo_actions,
                    batch_size=distill_batch_size,
                )
                distill_loss = F.cross_entropy(actor(demo_mb_obs), demo_mb_actions)
                loss = ppo_loss + float(cfg.distill_lambda) * distill_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params + critic_params, cfg.max_grad_norm)
                optimizer.step()
        ppo_update_count += 1

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
            demo_metrics = demonstration_metrics(actor, (demo_obs, demo_actions), device=device)
            elapsed = time.time() - start_time
            print(
                f"Steps={global_step} | meanR={mean_r:.1f} +/- {std_r:.1f} | "
                f"elapsed={elapsed:.1f}s | failure_rate={failure_rate:.2f} | "
                f"success_rate={success_rate:.2f} | "
                f"demo_acc={demo_metrics['source_demo_accuracy']:.3f} | "
                f"distill_lambda={cfg.distill_lambda}",
            )
            if early_stop_enabled and ppo_update_count >= cfg.early_stop_min_steps:
                reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
                    cfg,
                    mean_reward=mean_r,
                    failure_rate=failure_rate,
                    success_rate=success_rate,
                )
                if reward_ok and failure_ok and success_ok:
                    break

    if cfg.eval_episodes is not None and cfg.eval_episodes > 0:
        mean_r, std_r, failure_rate, success_rate = evaluate_with_success(
            env=eval_env,
            actor=actor,
            device=device,
            episodes=cfg.eval_episodes,
            deterministic=True,
        )
        print(
            f"Final evaluation over {cfg.eval_episodes} episodes: "
            f"mean_reward={mean_r:.2f} +/- {std_r:.2f} | "
            f"failure_rate={failure_rate:.2f} | success_rate={success_rate:.2f}",
        )

    env.close()
    if early_stop_eval_env is not None and early_stop_eval_env is not env:
        early_stop_eval_env.close()

    if return_training_data:
        training_data["states"] = np.array(training_data["states"])
        training_data["actions"] = np.array(training_data["actions"])
        training_data["terminated"] = np.array(training_data["terminated"])
        training_data["truncated"] = np.array(training_data["truncated"])
        training_data["safe"] = np.array(training_data["safe"])
        return actor, critic, training_data
    return actor, critic
