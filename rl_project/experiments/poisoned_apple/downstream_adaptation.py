#!/usr/bin/env python3
"""Downstream adaptation experiment for PoisonedApple.

Methods implemented on Task 2:
- UnsafeAdapt: unconstrained PPO adaptation
- EWC: PPO with Elastic Weight Consolidation regularisation
- SafeAdapt: PPO with Rashomon parameter bounds from Task-1 safety dataset
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import gymnasium as gym
from torch.utils.data import TensorDataset

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Plotting is optional; loaded lazily in plotting block.

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXP_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _EXP_DIR.parent.parent
_RL_DIR = _PROJECT_ROOT / "rl_project"
for p in (_SCRIPT_DIR, _EXP_DIR, _PROJECT_ROOT, _RL_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from poisoned_apple_env import (  # noqa: E402
    PoisonedAppleEnv,
    evaluate_policy,
    get_safety_critical_observations_and_safe_actions,
    visualize_agent_trajectory,
)
from rl_project.utils.ewc_ppo import EWCPPOConfig, compute_ewc_state, ewc_ppo_train  # noqa: E402
from rl_project.utils.ppo_utils import PPOConfig, make_actor_critic, ppo_train  # noqa: E402
from src.trainer.IntervalTrainer import IntervalTrainer  # noqa: E402


class AppendTaskIDObservationWrapper(gym.ObservationWrapper):
    """Append a constant task-id feature to 1D vector observations."""

    def __init__(self, env: gym.Env, task_id: int | float):
        super().__init__(env)
        self.task_id = float(task_id)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("AppendTaskIDObservationWrapper requires Box observation space.")
        if len(env.observation_space.shape) != 1:
            raise ValueError("AppendTaskIDObservationWrapper supports only 1D observations.")

        low = np.asarray(env.observation_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.observation_space.high, dtype=np.float32).reshape(-1)
        task_arr = np.asarray([self.task_id], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, task_arr], axis=0),
            high=np.concatenate([high, task_arr], axis=0),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        return np.concatenate([obs, np.asarray([self.task_id], dtype=np.float32)], axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream adaptation for PoisonedApple")
    p.add_argument("--cfg", type=str, default="simple_6x6", help="Config key in configs.yaml")
    p.add_argument("--seed", type=int, default=None, help="Override seed (default from cfg)")
    p.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help=(
            "Directory with source_policy.pt / source_critic.pt / source_training_data.pt, "
            "or run directory containing source/. Defaults to outputs/<cfg>/<seed>."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save downstream outputs (default: <run_dir>/downstream)",
    )
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Task-2 adaptation timesteps per method",
    )
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ewc-lambda", type=float, default=5_000.0)
    p.add_argument(
        "--ewc-fisher-sample-size",
        type=int,
        default=1_000,
        help="Max number of source states used for Fisher estimation",
    )
    p.add_argument("--rashomon-n-iters", type=int, default=20_000)
    p.add_argument(
        "--min-safety-accuracy",
        type=float,
        default=None,
        help="Hard min certificate threshold (0-1 or 0-100)",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save trajectory plots for each policy/task",
    )
    return p.parse_args()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _as_pos_tuple(raw: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field_name} must be a [row, col] pair, got {raw!r}")
    return int(raw[0]), int(raw[1])


def _as_positions(raw: Any, field_name: str) -> list[tuple[int, int]]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of [row, col] pairs")
    return [_as_pos_tuple(item, f"{field_name}[{idx}]") for idx, item in enumerate(raw)]


def _layout_from_cfg(cfg: dict[str, Any], task_id: int) -> dict[str, Any]:
    if task_id not in (1, 2):
        raise ValueError(f"task_id must be 1 or 2, got {task_id}")
    return {
        "agent_start_pos": _as_pos_tuple(cfg[f"task{task_id}_agent_start_pos"], f"task{task_id}_agent_start_pos"),
        "safe_apples": _as_positions(cfg[f"task{task_id}_safe_apples"], f"task{task_id}_safe_apples"),
        "poisoned_apples": _as_positions(cfg[f"task{task_id}_poisoned_apples"], f"task{task_id}_poisoned_apples"),
    }


def make_poisoned_apple_env(
    cfg: dict[str, Any],
    task_id: int,
    render_mode: str | None = None,
    append_task_id: bool = True,
) -> gym.Env:
    observation_type = str(cfg.get("observation_type", "flat"))
    if observation_type != "flat":
        raise ValueError(
            "downstream_adaptation.py currently supports observation_type='flat' only. "
            f"Got '{observation_type}'."
        )

    layout = _layout_from_cfg(cfg, task_id)
    env: gym.Env = PoisonedAppleEnv(
        grid_size=int(cfg["grid_size"]),
        observation_type=observation_type,
        max_steps=int(cfg.get("max_steps", 30)),
        reward_safe=float(cfg.get("reward_safe", 1.0)),
        reward_poison=float(cfg.get("reward_poison", -1.0)),
        reward_step=float(cfg.get("reward_step", -0.01)),
        agent_start_pos=layout["agent_start_pos"],
        safe_apple_positions=layout["safe_apples"],
        poisoned_apple_positions=layout["poisoned_apples"],
        render_mode=render_mode,
    )
    if append_task_id:
        env = AppendTaskIDObservationWrapper(env, task_id=task_id)
    return env


def _unwrap_poisoned_env(env: gym.Env) -> PoisonedAppleEnv:
    cur: gym.Env = env
    while isinstance(cur, gym.Wrapper):
        if isinstance(cur, PoisonedAppleEnv):
            return cur
        cur = cur.env
    if isinstance(cur, PoisonedAppleEnv):
        return cur
    raise TypeError("Expected PoisonedAppleEnv (possibly wrapped).")


def _align_observation_with_env(obs: np.ndarray, env: gym.Env) -> np.ndarray:
    """
    Align observation vector with env.observation_space dimension.

    If env uses AppendTaskIDObservationWrapper and obs is base-env vector,
    append the task id.
    """
    obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
    expected_dim = int(env.observation_space.shape[0])  # type: ignore[index]
    if obs_vec.size == expected_dim:
        return obs_vec
    if isinstance(env, AppendTaskIDObservationWrapper) and obs_vec.size + 1 == expected_dim:
        return np.concatenate(
            [obs_vec, np.asarray([env.task_id], dtype=np.float32)],
            axis=0,
        )
    raise ValueError(
        f"Observation dimension mismatch: got {obs_vec.size}, expected {expected_dim}."
    )


def _ensure_dataset_obs_dim(
    dataset: TensorDataset,
    expected_obs_dim: int,
    task_id: int,
) -> TensorDataset:
    """Make dataset observation width match actor/env width when task id is appended."""
    X, Y = dataset.tensors
    if X.ndim != 2:
        raise ValueError(f"Expected dataset X shape (N, D), got {tuple(X.shape)}.")
    if X.shape[1] == expected_obs_dim:
        return dataset
    if X.shape[1] + 1 == expected_obs_dim:
        task_col = torch.full((X.shape[0], 1), float(task_id), dtype=X.dtype)
        return TensorDataset(torch.cat([X, task_col], dim=1), Y)
    raise ValueError(
        f"Safety dataset observation dim mismatch: got {X.shape[1]}, expected {expected_obs_dim}."
    )


def _build_safety_rashomon_dataset(env_task1: gym.Env) -> TensorDataset:
    base_env = _unwrap_poisoned_env(env_task1)
    samples = get_safety_critical_observations_and_safe_actions(base_env, observation_type="flat")
    obs_dim = int(env_task1.observation_space.shape[0])  # type: ignore[index]

    if len(samples) == 0:
        X = torch.zeros((0, obs_dim), dtype=torch.float32)
        Y = torch.zeros((0, int(base_env.action_space.n)), dtype=torch.float32)
        return TensorDataset(X, Y)

    X_np = np.stack(
        [_align_observation_with_env(obs, env_task1) for obs, _ in samples],
        axis=0,
    ).astype(np.float32)
    Y_np = np.zeros((len(samples), int(base_env.action_space.n)), dtype=np.float32)
    for i, (_, safe_actions) in enumerate(samples):
        if len(safe_actions) == 0:
            raise ValueError(f"Safety sample {i} has no safe actions.")
        Y_np[i, safe_actions] = 1.0

    return TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(Y_np, dtype=torch.float32),
    )


def _critical_state_safety_rate(actor: torch.nn.Module, dataset: TensorDataset) -> float:
    if len(dataset) == 0:
        return 1.0
    X, Y = dataset.tensors
    actor.eval()
    with torch.no_grad():
        logits = actor(X)
        chosen_actions = torch.argmax(logits, dim=-1)
        is_safe = Y[torch.arange(Y.shape[0]), chosen_actions] > 0.5
    return float(is_safe.float().mean().item())


def _compute_rashomon_bounds(
    *,
    actor: torch.nn.Module,
    rashomon_dataset: TensorDataset,
    seed: int,
    rashomon_n_iters: int,
    min_hard_specification: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, float, int, float, int]:
    n_safe_per_state = rashomon_dataset.tensors[1].sum(dim=1).tolist()
    max_safe_actions_per_state = max(n_safe_per_state) if n_safe_per_state else 0
    min_surrogate_threshold = max_safe_actions_per_state / (1 + max_safe_actions_per_state)

    print(f"  Rashomon states: {len(n_safe_per_state)}")
    print(f"  Surrogate threshold: {min_surrogate_threshold:.6f}")

    actor = actor.cpu().eval()
    with torch.no_grad():
        all_obs = rashomon_dataset.tensors[0]
        safe_mask = rashomon_dataset.tensors[1]
        logits = actor(all_obs)

        inverse_temp = -1
        safe_prob_mass = None
        for t in range(10, 1001):
            probs = torch.softmax(logits * t, dim=1)
            safe_prob_mass = (probs * safe_mask).sum(dim=1)
            if safe_prob_mass.min().item() >= min_surrogate_threshold:
                inverse_temp = t
                break

        if inverse_temp == -1:
            worst_idx = int(safe_prob_mass.argmin().item())  # type: ignore[union-attr]
            raise RuntimeError(
                "Could not find inverse_temp <= 1000 that satisfies surrogate threshold. "
                f"Worst idx={worst_idx}, mass={safe_prob_mass[worst_idx].item():.6f}, "  # type: ignore[index]
                f"required={min_surrogate_threshold:.6f}."
            )

    print(f"  Smallest inverse_temp: {inverse_temp}")
    print(f"  Min safe-action softmax mass at T={inverse_temp}: {safe_prob_mass.min().item():.6f}")  # type: ignore[union-attr]

    _set_seeds(seed)
    interval_trainer = IntervalTrainer(
        model=actor,
        min_acc_limit=min_surrogate_threshold,
        min_acc_increment=0.0,
        seed=seed,
        n_iters=rashomon_n_iters,
        T=inverse_temp,
        checkpoint=100,
    )
    interval_trainer.compute_rashomon_set(
        dataset=rashomon_dataset,
        multi_label=True,
        aggregation="min",
    )

    valid_idxs = [
        i for i, cert in enumerate(interval_trainer.certificates)
        if cert >= min_hard_specification
    ]
    if not valid_idxs:
        best = float(max(interval_trainer.certificates))
        raise RuntimeError(
            "No Rashomon certificate satisfies required hard threshold "
            f"({min_hard_specification:.4f}). Best certificate={best:.4f}."
        )

    final_idx = int(valid_idxs[-1])
    selected_cert = float(interval_trainer.certificates[final_idx])
    print(f"  Selected hard certificate: {selected_cert:.4f}")

    bounded_model = interval_trainer.bounds[final_idx]
    param_bounds_l = [p.detach().cpu() for p in bounded_model.param_l]
    param_bounds_u = [p.detach().cpu() for p in bounded_model.param_u]

    return (
        param_bounds_l,
        param_bounds_u,
        bounded_model,
        selected_cert,
        final_idx,
        float(min_surrogate_threshold),
        int(inverse_temp),
    )


def _train_unsafeadapt(
    *,
    env: PoisonedAppleEnv,
    actor_init: torch.nn.Sequential,
    critic_init: torch.nn.Sequential,
    seed: int,
    total_timesteps: int,
    ent_coef: float,
    eval_episodes: int,
    device: str,
    early_stop_total_reward_threshold: float | None = None,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    cfg = PPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        eval_episodes=eval_episodes,
        ent_coef=ent_coef,
        early_stop=(early_stop_total_reward_threshold is not None),
        early_stop_min_steps=0,
        early_stop_reward_threshold=early_stop_total_reward_threshold,
        device=device,
    )
    actor, critic = ppo_train(
        env=env,
        cfg=cfg,
        actor_warm_start=copy.deepcopy(actor_init),
        critic_warm_start=copy.deepcopy(critic_init),
    )
    return actor.cpu(), critic.cpu()


def _train_safeadapt(
    *,
    env: PoisonedAppleEnv,
    actor_init: torch.nn.Sequential,
    critic_init: torch.nn.Sequential,
    seed: int,
    total_timesteps: int,
    ent_coef: float,
    eval_episodes: int,
    device: str,
    actor_param_bounds_l: list[torch.Tensor],
    actor_param_bounds_u: list[torch.Tensor],
    early_stop_total_reward_threshold: float | None = None,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    cfg = PPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        eval_episodes=eval_episodes,
        ent_coef=ent_coef,
        early_stop=(early_stop_total_reward_threshold is not None),
        early_stop_min_steps=0,
        early_stop_reward_threshold=early_stop_total_reward_threshold,
        device=device,
    )
    actor, critic = ppo_train(
        env=env,
        cfg=cfg,
        actor_warm_start=copy.deepcopy(actor_init),
        critic_warm_start=copy.deepcopy(critic_init),
        actor_param_bounds_l=actor_param_bounds_l,
        actor_param_bounds_u=actor_param_bounds_u,
    )
    return actor.cpu(), critic.cpu()


def _train_ewc(
    *,
    env: PoisonedAppleEnv,
    actor_init: torch.nn.Sequential,
    critic_init: torch.nn.Sequential,
    ewc_states: list,
    seed: int,
    total_timesteps: int,
    ent_coef: float,
    eval_episodes: int,
    device: str,
    ewc_lambda: float,
    early_stop_total_reward_threshold: float | None = None,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    cfg = EWCPPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        eval_episodes=eval_episodes,
        ent_coef=ent_coef,
        early_stop=(early_stop_total_reward_threshold is not None),
        early_stop_min_steps=0,
        early_stop_reward_threshold=early_stop_total_reward_threshold,
        device=device,
        ewc_lambda=ewc_lambda,
        ewc_apply_to_critic=False,
    )
    actor, critic = ewc_ppo_train(
        env=env,
        cfg=cfg,
        ewc_states=ewc_states,
        actor_warm_start=copy.deepcopy(actor_init),
        critic_warm_start=copy.deepcopy(critic_init),
    )
    return actor.cpu(), critic.cpu()


def _evaluate_policy_on_task(
    cfg: dict[str, Any],
    task_id: int,
    actor: torch.nn.Module,
    num_episodes: int,
    append_task_id: bool,
) -> dict[str, float]:
    env = make_poisoned_apple_env(
        cfg,
        task_id=task_id,
        render_mode=None,
        append_task_id=append_task_id,
    )
    metrics = evaluate_policy(env, actor, num_episodes=num_episodes)
    env.close()
    return metrics


def _plot_policy_trajectories(
    *,
    cfg: dict[str, Any],
    policies: dict[str, torch.nn.Module],
    seed: int,
    save_dir: Path,
    append_task_id: bool,
) -> None:
    try:
        for policy_name, actor in policies.items():
            for task_id in (1, 2):
                env = make_poisoned_apple_env(
                    cfg,
                    task_id=task_id,
                    render_mode=None,
                    append_task_id=append_task_id,
                )
                visualize_agent_trajectory(
                    env=env,
                    actor=actor,
                    num_episodes=1,
                    env_name=f"Task_{task_id}",
                    cfg_name=cfg,
                    actor_name=policy_name,
                    save_dir=str(save_dir),
                )
                env.close()
    except ModuleNotFoundError as exc:
        print(f"  Skipping plots due to missing optional dependency: {exc}")


def main() -> None:
    args = parse_args()

    config_path = _SCRIPT_DIR / "configs.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        all_cfgs = yaml.safe_load(f)
    if args.cfg not in all_cfgs:
        raise ValueError(f"Config '{args.cfg}' not found. Available: {list(all_cfgs)}")

    cfg = all_cfgs[args.cfg]
    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    total_timesteps = int(
        cfg.get("downstream_total_timesteps", cfg.get("source_total_timesteps", 30_000))
        if args.total_timesteps is None
        else args.total_timesteps
    )
    eval_episodes = int(cfg.get("eval_episodes", 100) if args.eval_episodes is None else args.eval_episodes)
    eval_episodes = max(1, eval_episodes)

    min_safety_accuracy = cfg.get("min_safety_accuracy", 1.0)
    if args.min_safety_accuracy is not None:
        min_safety_accuracy = args.min_safety_accuracy
    min_safety_accuracy = float(min_safety_accuracy)
    if min_safety_accuracy > 1.0:
        min_safety_accuracy = min_safety_accuracy / 100.0
    append_task_id = bool(cfg.get("append_task_id", True))
    task2_early_stop_reward_threshold = cfg.get("task2_max_total_reward", None)
    if task2_early_stop_reward_threshold is not None:
        task2_early_stop_reward_threshold = float(task2_early_stop_reward_threshold)

    _set_seeds(seed)

    # Paths
    if args.source_dir:
        source_path = Path(args.source_dir)
        if (source_path / "source_policy.pt").exists():
            source_dir = source_path
            run_dir = source_path.parent
        else:
            run_dir = source_path
            source_dir = run_dir / "source"
    else:
        run_dir = _SCRIPT_DIR / "outputs" / args.cfg / str(seed)
        source_dir = run_dir / "source"

    downstream_dir = Path(args.output_dir) if args.output_dir else (run_dir / "downstream")
    downstream_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots" if args.output_dir is None else (downstream_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DOWNSTREAM ADAPTATION EXPERIMENT (PoisonedApple)")
    print("=" * 80)
    print(f"Config         : {args.cfg}")
    print(f"Seed           : {seed}")
    print(f"Source dir     : {source_dir}")
    print(f"Run dir        : {run_dir}")
    print(f"Downstream dir : {downstream_dir}")
    print(f"Timesteps      : {total_timesteps}")
    print(f"Eval episodes  : {eval_episodes}")
    if task2_early_stop_reward_threshold is not None:
        print(f"Early stop     : enabled (Task-2 eval total reward >= {task2_early_stop_reward_threshold:.4f})")
    else:
        print("Early stop     : disabled (task2_max_total_reward not provided in config)")
    print(f"EWC lambda     : {args.ewc_lambda}")
    print(f"Append task id : {append_task_id}")
    print(f"Device         : {args.device}")
    print("=" * 80)

    # 1) Load source policy and artifacts
    print("\n[1/7] Loading source policy, critic, and training data ...")
    env_tmp = make_poisoned_apple_env(
        cfg,
        task_id=1,
        render_mode=None,
        append_task_id=append_task_id,
    )
    obs_dim = int(env_tmp.observation_space.shape[0])  # type: ignore[index]
    n_actions = int(env_tmp.action_space.n)  # type: ignore[union-attr]
    env_tmp.close()

    source_actor, source_critic, _ = make_actor_critic(obs_dim=obs_dim, n_actions=n_actions)
    source_actor.load_state_dict(torch.load(source_dir / "source_policy.pt", map_location="cpu"))
    source_critic.load_state_dict(torch.load(source_dir / "source_critic.pt", map_location="cpu"))

    source_training_data = torch.load(
        source_dir / "source_training_data.pt",
        map_location="cpu",
        weights_only=False,
    )
    source_train_states = np.asarray(source_training_data.get("states", []), dtype=np.float32)
    if source_train_states.shape[0] == 0:
        raise RuntimeError("source_training_data.pt contains no states; cannot run EWC.")
    print(f"  Source training transitions: {source_train_states.shape[0]}")

    # 2) Build/load Task-1 safety dataset and compute Rashomon bounds
    print("\n[2/7] Building safety Rashomon dataset and computing bounds ...")
    safety_dataset_path = source_dir / "task1_safety_dataset.pt"
    if safety_dataset_path.exists():
        loaded = torch.load(safety_dataset_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "X" in loaded and "Y_multi_hot" in loaded:
            safety_dataset = TensorDataset(
                torch.as_tensor(loaded["X"], dtype=torch.float32),
                torch.as_tensor(loaded["Y_multi_hot"], dtype=torch.float32),
            )
        elif isinstance(loaded, TensorDataset):
            safety_dataset = loaded
        else:
            raise ValueError(
                f"Unsupported safety dataset artifact format at {safety_dataset_path}."
            )
        print(f"  Loaded safety dataset from source artifacts: {len(safety_dataset)} states")
    else:
        env_t1_for_dataset = make_poisoned_apple_env(
            cfg,
            task_id=1,
            render_mode=None,
            append_task_id=append_task_id,
        )
        env_t1_for_dataset.reset(seed=seed)
        safety_dataset = _build_safety_rashomon_dataset(env_t1_for_dataset)
        env_t1_for_dataset.close()
        print(f"  Built safety dataset from Task-1 layout: {len(safety_dataset)} states")

    if len(safety_dataset) == 0:
        raise RuntimeError("Safety dataset is empty; SafeAdapt cannot be certified.")
    safety_dataset = _ensure_dataset_obs_dim(
        safety_dataset,
        expected_obs_dim=obs_dim,
        task_id=1,
    )

    source_safety_acc = _critical_state_safety_rate(source_actor, safety_dataset)
    print(f"  Source critical-state safety rate: {source_safety_acc:.4f}")

    (
        safe_param_bounds_l,
        safe_param_bounds_u,
        safe_bounded_model,
        selected_cert,
        selected_bound_index,
        surrogate_threshold,
        inverse_temp,
    ) = _compute_rashomon_bounds(
        actor=copy.deepcopy(source_actor),
        rashomon_dataset=safety_dataset,
        seed=seed,
        rashomon_n_iters=args.rashomon_n_iters,
        min_hard_specification=min_safety_accuracy,
    )

    torch.save(safety_dataset, downstream_dir / "rashomon_dataset_safety.pt")
    torch.save(safety_dataset, downstream_dir / "rashomon_dataset.pt")  # backward compatible
    torch.save(safe_bounded_model, downstream_dir / "bounded_model_safety.pt")
    torch.save(safe_bounded_model, downstream_dir / "bounded_model.pt")  # backward compatible

    # 3) SafeAdapt (bounded PPO)
    print("\n[3/7] SafeAdapt: PPO with Rashomon parameter bounds ...")
    env_t2_safe = make_poisoned_apple_env(
        cfg,
        task_id=2,
        render_mode=None,
        append_task_id=append_task_id,
    )
    safeadapt_actor, safeadapt_critic = _train_safeadapt(
        env=env_t2_safe,
        actor_init=source_actor,
        critic_init=source_critic,
        seed=seed,
        total_timesteps=total_timesteps,
        ent_coef=args.ent_coef,
        eval_episodes=eval_episodes,
        device=args.device,
        actor_param_bounds_l=safe_param_bounds_l,
        actor_param_bounds_u=safe_param_bounds_u,
        early_stop_total_reward_threshold=task2_early_stop_reward_threshold,
    )

    # 4) UnsafeAdapt (unconstrained PPO)
    print("\n[4/7] UnsafeAdapt: unconstrained PPO adaptation ...")
    env_t2_unsafe = make_poisoned_apple_env(
        cfg,
        task_id=2,
        render_mode=None,
        append_task_id=append_task_id,
    )
    unsafeadapt_actor, unsafeadapt_critic = _train_unsafeadapt(
        env=env_t2_unsafe,
        actor_init=source_actor,
        critic_init=source_critic,
        seed=seed,
        total_timesteps=total_timesteps,
        ent_coef=args.ent_coef,
        eval_episodes=eval_episodes,
        device=args.device,
        early_stop_total_reward_threshold=task2_early_stop_reward_threshold,
    )

    # 5) EWC adaptation
    print("\n[5/7] EWC: computing Fisher and adapting with EWC PPO ...")
    ewc_state = compute_ewc_state(
        actor=copy.deepcopy(source_actor),
        observations=source_train_states,
        compute_critic=False,
        device=args.device,
        fisher_sample_size=min(args.ewc_fisher_sample_size, source_train_states.shape[0]),
        seed=seed,
    )

    env_t2_ewc = make_poisoned_apple_env(
        cfg,
        task_id=2,
        render_mode=None,
        append_task_id=append_task_id,
    )
    ewc_actor, ewc_critic = _train_ewc(
        env=env_t2_ewc,
        actor_init=source_actor,
        critic_init=source_critic,
        ewc_states=[ewc_state],
        seed=seed,
        total_timesteps=total_timesteps,
        ent_coef=args.ent_coef,
        eval_episodes=eval_episodes,
        device=args.device,
        ewc_lambda=args.ewc_lambda,
        early_stop_total_reward_threshold=task2_early_stop_reward_threshold,
    )

    # 6) Evaluate all policies on Task 1 and Task 2
    print("\n[6/7] Evaluating policies on Task 1 and Task 2 ...")
    policies: dict[str, torch.nn.Module] = {
        "Source": source_actor.cpu(),
        "SafeAdapt": safeadapt_actor.cpu(),
        "UnsafeAdapt": unsafeadapt_actor.cpu(),
        "EWC": ewc_actor.cpu(),
    }

    rows: list[dict[str, Any]] = []
    for policy_name, actor in policies.items():
        actor.eval()
        for task_id in (1, 2):
            metrics = _evaluate_policy_on_task(
                cfg=cfg,
                task_id=task_id,
                actor=actor,
                num_episodes=eval_episodes,
                append_task_id=append_task_id,
            )

            critical_state_safety = None
            if task_id == 1:
                critical_state_safety = _critical_state_safety_rate(actor, safety_dataset)

            rows.append(
                {
                    "Policy": policy_name,
                    "Task": task_id,
                    "Trajectory Safety Rate": float(metrics["avg_safety_success"]),
                    "Critical State Safety Rate": critical_state_safety,
                    "Avg Total Reward": float(metrics["avg_reward"]),
                    "Avg Performance Success": float(metrics["avg_performance_success"]),
                    "Avg Overall Success": float(metrics["avg_overall_success"]),
                }
            )

    results_df = pd.DataFrame(rows)
    results_path = downstream_dir / "results_table.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nTable saved to {results_path}")

    # 7) Save artifacts
    print("\n[7/7] Saving checkpoints, summary, and plots ...")
    torch.save(safeadapt_actor.state_dict(), downstream_dir / "safeadapt_actor.pt")
    torch.save(unsafeadapt_actor.state_dict(), downstream_dir / "unsafeadapt_actor.pt")
    torch.save(ewc_actor.state_dict(), downstream_dir / "ewc_actor.pt")

    # Optional critics
    torch.save(safeadapt_critic.state_dict(), downstream_dir / "safeadapt_critic.pt")
    torch.save(unsafeadapt_critic.state_dict(), downstream_dir / "unsafeadapt_critic.pt")
    torch.save(ewc_critic.state_dict(), downstream_dir / "ewc_critic.pt")

    summary_payload = {
        "cfg": args.cfg,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "eval_episodes": eval_episodes,
        "ent_coef": args.ent_coef,
        "ewc_lambda": args.ewc_lambda,
        "ewc_fisher_sample_size": int(min(args.ewc_fisher_sample_size, source_train_states.shape[0])),
        "rashomon_n_iters": args.rashomon_n_iters,
        "min_safety_accuracy": min_safety_accuracy,
        "append_task_id": append_task_id,
        "task2_early_stop_reward_threshold": task2_early_stop_reward_threshold,
        "source_safety_accuracy": source_safety_acc,
        "selected_certificate": selected_cert,
        "selected_bound_index": selected_bound_index,
        "surrogate_threshold": surrogate_threshold,
        "inverse_temp": inverse_temp,
        "source_dir": str(source_dir),
        "downstream_dir": str(downstream_dir),
    }
    with (downstream_dir / "downstream_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    if args.save_plots:
        _plot_policy_trajectories(
            cfg=cfg,
            policies=policies,
            seed=seed,
            save_dir=plots_dir,
            append_task_id=append_task_id,
        )

    print(f"\nAll downstream artifacts saved to: {downstream_dir}")
    if args.save_plots:
        print(f"All plots saved to: {plots_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
