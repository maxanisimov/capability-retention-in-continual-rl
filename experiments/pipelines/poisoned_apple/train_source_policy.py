#!/usr/bin/env python3
"""Train a PoisonedApple source policy (Task 1) and safety-finetune it with BC.

Workflow:
1. Train PPO on Task 1 to get a high-performing source policy.
2. Build a safety dataset over safety-critical states (multi-label safe actions).
3. Finetune the actor with multi-label behavior cloning.
4. Verify the finetuned actor is safe over all states in Task 1 layout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # poisoned_apple/
_EXP_DIR = _SCRIPT_DIR.parent                          # experiments/
_PROJECT_ROOT = _EXP_DIR.parent.parent                 # CertifiedContinualLearning/
_RL_DIR = _PROJECT_ROOT / "experiments"
for p in (_EXP_DIR, _RL_DIR, _PROJECT_ROOT, _SCRIPT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from poisoned_apple_env import (  # noqa: E402
    PoisonedAppleEnv,
    evaluate_policy,
    get_observation,
    get_safety_critical_observations_and_safe_actions,
    visualize_agent_trajectory,
)
from experiments.utils.ppo_utils import PPOConfig, ppo_train  # noqa: E402

N_ACTIONS = 4


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


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _as_pos_tuple(raw: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{field_name} must be a 2D [row, col] pair. Got: {raw!r}")
    return int(raw[0]), int(raw[1])


def _as_positions(raw: Any, field_name: str) -> list[tuple[int, int]]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of [row, col] entries.")
    return [_as_pos_tuple(p, f"{field_name}[{i}]") for i, p in enumerate(raw)]


def _layout_from_cfg(cfg: dict[str, Any], task_id: int) -> dict[str, Any]:
    if task_id not in (1, 2):
        raise ValueError(f"task_id must be 1 or 2, got {task_id}.")
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
    layout = _layout_from_cfg(cfg, task_id)
    observation_type = str(cfg.get("observation_type", "flat"))
    if observation_type != "flat":
        raise ValueError(
            "This script currently supports observation_type='flat' only, "
            f"got '{observation_type}'."
        )

    env: gym.Env = PoisonedAppleEnv(
        grid_size=int(cfg["grid_size"]),
        observation_type=observation_type,
        agent_start_pos=layout["agent_start_pos"],
        safe_apple_positions=layout["safe_apples"],
        poisoned_apple_positions=layout["poisoned_apples"],
        max_steps=int(cfg.get("max_steps", 30)),
        reward_safe=float(cfg.get("reward_safe", 1.0)),
        reward_poison=float(cfg.get("reward_poison", -1.0)),
        reward_step=float(cfg.get("reward_step", -0.01)),
        render_mode=render_mode,
    )
    if append_task_id:
        env = AppendTaskIDObservationWrapper(env, task_id=task_id)
    return env


def _unwrap_poisoned_env(env: gym.Env) -> PoisonedAppleEnv:
    """Extract the base PoisonedAppleEnv from wrappers."""
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
    Return observation aligned with env.observation_space shape.

    If env uses AppendTaskIDObservationWrapper and obs is base-env flat vector,
    append the task id automatically.
    """
    obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
    expected_dim = int(env.observation_space.shape[0])  # type: ignore[index]
    if obs_vec.size == expected_dim:
        return obs_vec

    if (
        isinstance(env, AppendTaskIDObservationWrapper)
        and obs_vec.size + 1 == expected_dim
    ):
        return np.concatenate(
            [obs_vec, np.asarray([env.task_id], dtype=np.float32)],
            axis=0,
        )

    raise ValueError(
        f"Observation dimension mismatch: got {obs_vec.size}, expected {expected_dim}."
    )


def _make_ppo_cfg(
    cfg: dict[str, Any],
    seed: int,
    total_steps: int,
    eval_episodes: int,
    device: str,
) -> PPOConfig:
    return PPOConfig(
        seed=seed,
        total_timesteps=int(total_steps),
        eval_episodes=max(1, int(eval_episodes)),
        rollout_steps=int(cfg.get("ppo_rollout_steps", 256)),
        update_epochs=int(cfg.get("ppo_update_epochs", 6)),
        minibatch_size=int(cfg.get("ppo_minibatch_size", 64)),
        gamma=float(cfg.get("ppo_gamma", 0.99)),
        gae_lambda=float(cfg.get("ppo_gae_lambda", 0.95)),
        clip_coef=float(cfg.get("ppo_clip_coef", 0.2)),
        ent_coef=float(cfg.get("ppo_ent_coef", 0.01)),
        vf_coef=float(cfg.get("ppo_vf_coef", 0.5)),
        lr=float(cfg.get("ppo_lr", 3e-4)),
        max_grad_norm=float(cfg.get("ppo_max_grad_norm", 0.5)),
        device=device,
    )


def _next_position(row: int, col: int, action: int, grid_size: int) -> tuple[int, int]:
    if action == PoisonedAppleEnv.UP:
        return max(0, row - 1), col
    if action == PoisonedAppleEnv.RIGHT:
        return row, min(grid_size - 1, col + 1)
    if action == PoisonedAppleEnv.DOWN:
        return min(grid_size - 1, row + 1), col
    if action == PoisonedAppleEnv.LEFT:
        return row, max(0, col - 1)
    raise ValueError(f"Unknown action id: {action}")


def _build_safety_dataset(
    env: gym.Env,
) -> tuple[TensorDataset, list[tuple[np.ndarray, list[int]]]]:
    """Create TensorDataset(X, Y_multi_hot) for safety-critical states."""
    base_env = _unwrap_poisoned_env(env)
    samples = get_safety_critical_observations_and_safe_actions(base_env, observation_type="flat")

    if len(samples) == 0:
        obs_dim = int(env.observation_space.shape[0])  # type: ignore[index]
        X = torch.zeros((0, obs_dim), dtype=torch.float32)
        Y = torch.zeros((0, N_ACTIONS), dtype=torch.float32)
        return TensorDataset(X, Y), samples

    X_np = np.stack(
        [_align_observation_with_env(obs, env) for obs, _ in samples],
        axis=0,
    ).astype(np.float32)
    Y_np = np.zeros((len(samples), N_ACTIONS), dtype=np.float32)

    for i, (_, safe_actions) in enumerate(samples):
        if len(safe_actions) == 0:
            raise ValueError(f"State index {i} has no safe actions; cannot build safety dataset.")
        Y_np[i, safe_actions] = 1.0

    X = torch.tensor(X_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)
    return TensorDataset(X, Y), samples


def _safety_action_accuracy(actor: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    """Argmax-action safety accuracy for a multi-label safe-action target."""
    if X.numel() == 0:
        return 1.0
    actor.eval()
    with torch.no_grad():
        logits = actor(X)
        preds = torch.argmax(logits, dim=-1)
        safe = Y[torch.arange(Y.shape[0]), preds] > 0.5
    return float(safe.float().mean().item())


def finetune_policy_with_safety_bc(
    policy: torch.nn.Module,
    dataset: TensorDataset,
    required_accuracy: float,
    lr: float = 2e-3,
    max_epochs: int = 2_000,
    batch_size: int = 64,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Finetune policy with multi-label BC (BCE logits) on safety dataset."""
    if required_accuracy > 1.0:
        required_accuracy = required_accuracy / 100.0
    required_accuracy = float(required_accuracy)

    if not hasattr(dataset, "tensors") or len(dataset.tensors) != 2:
        raise ValueError("Expected TensorDataset(X, Y_multi_hot).")

    X, Y = dataset.tensors
    if Y.ndim != 2:
        raise ValueError(f"Expected Y to have shape (N, A), got {tuple(Y.shape)}.")

    if X.shape[0] == 0:
        if verbose:
            print("  Safety dataset is empty; skipping finetuning.")
        return {
            "final_accuracy": 1.0,
            "target_accuracy": required_accuracy,
            "epochs_run": 0,
            "reached_target": True,
        }

    torch.manual_seed(seed)

    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=True,
    )

    init_acc = _safety_action_accuracy(policy, X, Y)
    if verbose:
        print("\n--- Safety BC finetuning ---")
        print(f"  dataset states: {len(dataset)}")
        print(f"  initial safety-action accuracy: {init_acc:.3f}")
        print(f"  target accuracy: {required_accuracy:.3f}")

    if init_acc >= required_accuracy:
        policy.eval()
        return {
            "final_accuracy": init_acc,
            "target_accuracy": required_accuracy,
            "epochs_run": 0,
            "reached_target": True,
        }

    reached = False
    final_acc = init_acc
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        policy.train()
        epoch_loss = 0.0
        batch_count = 0
        for xb, yb in loader:
            logits = policy(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1

        epochs_run = epoch
        final_acc = _safety_action_accuracy(policy, X, Y)

        if verbose and (epoch % 100 == 0 or final_acc >= required_accuracy):
            avg_loss = epoch_loss / max(batch_count, 1)
            print(
                f"  Epoch {epoch:4d} | loss={avg_loss:.6f} | "
                f"safety-action-acc={final_acc:.3f}"
            )

        if final_acc >= required_accuracy:
            reached = True
            break

    policy.eval()
    return {
        "final_accuracy": float(final_acc),
        "target_accuracy": required_accuracy,
        "epochs_run": epochs_run,
        "reached_target": reached,
    }


def verify_global_state_safety(
    env: gym.Env,
    actor: torch.nn.Module,
) -> dict[str, Any]:
    """Check chosen action safety for all possible agent positions on fixed layout."""
    base_env = _unwrap_poisoned_env(env)
    if base_env.safe_apples is None or base_env.poisoned_apples is None:
        raise ValueError("Environment must be reset before global safety verification.")

    poisoned_positions = set(base_env.poisoned_apples)
    safe_positions = set(base_env.safe_apples)

    unsafe_examples: list[dict[str, Any]] = []
    total_states = base_env.grid_size * base_env.grid_size
    safe_count = 0

    actor.eval()
    with torch.no_grad():
        for row in range(base_env.grid_size):
            for col in range(base_env.grid_size):
                obs = get_observation(
                    agent_position=(row, col),
                    safe_apple_positions=safe_positions,
                    poisoned_apple_positions=poisoned_positions,
                    grid_size=base_env.grid_size,
                    observation_type="flat",
                )
                obs_model = _align_observation_with_env(obs, env)
                obs_t = torch.as_tensor(obs_model, dtype=torch.float32).unsqueeze(0)
                action = int(torch.argmax(actor(obs_t), dim=-1).item())
                next_pos = _next_position(row, col, action, base_env.grid_size)
                is_safe = next_pos not in poisoned_positions
                if is_safe:
                    safe_count += 1
                elif len(unsafe_examples) < 12:
                    unsafe_examples.append(
                        {
                            "state": [row, col],
                            "action": action,
                            "next_pos": [int(next_pos[0]), int(next_pos[1])],
                        }
                    )

    accuracy = float(safe_count / max(total_states, 1))
    return {
        "safe_states": safe_count,
        "total_states": total_states,
        "safety_accuracy": accuracy,
        "unsafe_examples": unsafe_examples,
    }


def _print_metrics(header: str, metrics: dict[str, float]) -> None:
    print(header)
    print(f"  avg_reward:              {metrics['avg_reward']:.3f}")
    print(f"  avg_performance_success: {metrics['avg_performance_success']:.3f}")
    print(f"  avg_safety_success:      {metrics['avg_safety_success']:.3f}")
    print(f"  avg_overall_success:     {metrics['avg_overall_success']:.3f}")


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO source policy on PoisonedApple Task 1 and finetune with "
            "multi-label safety behavior cloning."
        )
    )
    parser.add_argument("--cfg", type=str, default="simple_6x6", help="Config key in configs.yaml.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed (default from cfg).")
    parser.add_argument("--total-steps", type=int, default=None, help="PPO training timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes.")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Torch device for PPO training.",
    )
    parser.add_argument(
        "--safety-finetuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run safety BC finetuning.",
    )
    parser.add_argument("--bc-epochs", type=int, default=None, help="Max BC finetune epochs.")
    parser.add_argument("--bc-lr", type=float, default=None, help="BC learning rate.")
    parser.add_argument("--bc-batch-size", type=int, default=None, help="BC batch size.")
    parser.add_argument(
        "--min-safety-accuracy",
        type=float,
        default=None,
        help="Required BC/global safety accuracy (0-1 or 0-100).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save outputs (default: outputs/<cfg>/<seed>/source).",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory for plots (default: outputs/<cfg>/<seed>/plots).",
    )
    args = parser.parse_args()

    cfg_path = _SCRIPT_DIR / "configs.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        all_cfgs = yaml.safe_load(f)

    if args.cfg not in all_cfgs:
        raise ValueError(f"Config '{args.cfg}' not found in {cfg_path}. Available: {list(all_cfgs)}")
    cfg = all_cfgs[args.cfg]

    seed = int(cfg.get("seed", 0) if args.seed is None else args.seed)
    total_steps = int(cfg.get("source_total_timesteps", 50_000) if args.total_steps is None else args.total_steps)
    eval_episodes = int(cfg.get("eval_episodes", 100) if args.eval_episodes is None else args.eval_episodes)

    bc_epochs = int(cfg.get("finetune_epochs", 2_000) if args.bc_epochs is None else args.bc_epochs)
    bc_lr = float(cfg.get("finetune_lr", 2e-3) if args.bc_lr is None else args.bc_lr)
    bc_batch_size = int(cfg.get("finetune_batch_size", 64) if args.bc_batch_size is None else args.bc_batch_size)

    required_safety_acc = cfg.get("min_safety_accuracy", 1.0)
    if args.min_safety_accuracy is not None:
        required_safety_acc = args.min_safety_accuracy
    required_safety_acc = float(required_safety_acc)
    if required_safety_acc > 1.0:
        required_safety_acc = required_safety_acc / 100.0

    required_task1_overall_success = float(cfg.get("required_task1_overall_success", 0.95))
    append_task_id = bool(cfg.get("append_task_id", True))

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (_SCRIPT_DIR / "outputs" / args.cfg / str(seed) / "source")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = (
        Path(args.plots_dir)
        if args.plots_dir is not None
        else (out_dir.parent / "plots")
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    _set_seeds(seed)

    print("=" * 72)
    print("PoisonedApple Source Training + Safety BC")
    print("=" * 72)
    print(f"  Config:            {args.cfg}")
    print(f"  Seed:              {seed}")
    print(f"  PPO timesteps:     {total_steps}")
    print(f"  Eval episodes:     {eval_episodes}")
    print(f"  PPO device:        {args.device}")
    print(f"  Append task id:    {append_task_id}")
    print(f"  Safety finetuning: {args.safety_finetuning}")
    print(f"  Output dir:        {out_dir}")
    print(f"  Plots dir:         {plots_dir}")

    # ── Step 1: train source policy on Task 1 ─────────────────────────────
    print("\n[1/4] Training source PPO policy on Task 1 ...")
    train_env = make_poisoned_apple_env(
        cfg,
        task_id=1,
        render_mode=None,
        append_task_id=append_task_id,
    )
    ppo_cfg = _make_ppo_cfg(
        cfg=cfg,
        seed=seed,
        total_steps=total_steps,
        eval_episodes=eval_episodes,
        device=args.device,
    )

    actor, critic, training_data = ppo_train(
        env=train_env,
        cfg=ppo_cfg,
        return_training_data=True,
    )
    train_env.close()

    # Keep actor/critic on CPU for env helpers in poisoned_apple_env.py
    actor = actor.cpu()
    critic = critic.cpu()

    # ── Step 2: evaluate source policy ─────────────────────────────────────
    print("\n[2/4] Evaluating source policy ...")
    eval_env_t1 = make_poisoned_apple_env(
        cfg,
        task_id=1,
        render_mode=None,
        append_task_id=append_task_id,
    )
    eval_env_t2 = make_poisoned_apple_env(
        cfg,
        task_id=2,
        render_mode=None,
        append_task_id=append_task_id,
    )

    source_metrics_t1 = evaluate_policy(eval_env_t1, actor, num_episodes=eval_episodes)
    source_metrics_t2 = evaluate_policy(eval_env_t2, actor, num_episodes=eval_episodes)

    _print_metrics("Task 1 source metrics:", source_metrics_t1)
    _print_metrics("Task 2 source metrics:", source_metrics_t2)

    if source_metrics_t1["avg_overall_success"] < required_task1_overall_success:
        raise RuntimeError(
            "Source PPO policy did not reach required Task-1 performance. "
            f"avg_overall_success={source_metrics_t1['avg_overall_success']:.3f}, "
            f"required>={required_task1_overall_success:.3f}. "
            "Try increasing source_total_timesteps."
        )

    # Build fixed-layout reset for safety dataset + verification.
    _ = eval_env_t1.reset(seed=seed)

    finetune_result: dict[str, Any] | None = None
    safety_dataset: TensorDataset | None = None
    # ── Step 3: safety BC finetuning ───────────────────────────────────────
    if args.safety_finetuning:
        print("\n[3/4] Building safety dataset and finetuning with BC ...")
        safety_dataset, _ = _build_safety_dataset(eval_env_t1)

        print(f"  Safety-critical states: {len(safety_dataset)}")
        if len(safety_dataset) > 0:
            X_s, Y_s = safety_dataset.tensors
            init_safety_acc = _safety_action_accuracy(actor, X_s, Y_s)
            print(f"  Pre-finetune safety-action accuracy: {init_safety_acc:.3f}")

        finetune_result = finetune_policy_with_safety_bc(
            policy=actor,
            dataset=safety_dataset,
            required_accuracy=required_safety_acc,
            lr=bc_lr,
            max_epochs=bc_epochs,
            batch_size=bc_batch_size,
            seed=seed,
            verbose=True,
        )

        if not finetune_result["reached_target"]:
            raise RuntimeError(
                "Safety BC did not reach required safety accuracy. "
                f"Final={finetune_result['final_accuracy']:.3f}, "
                f"required={finetune_result['target_accuracy']:.3f}."
            )

    # ── Step 4: verify global safety + save artifacts ──────────────────────
    print("\n[4/4] Verifying global safety and saving outputs ...")
    global_safety = verify_global_state_safety(eval_env_t1, actor)
    print(
        "  Global state safety: "
        f"{global_safety['safe_states']}/{global_safety['total_states']} "
        f"({global_safety['safety_accuracy']:.3f})"
    )

    if global_safety["safety_accuracy"] < required_safety_acc:
        raise RuntimeError(
            "Global safety verification failed. "
            f"safety_accuracy={global_safety['safety_accuracy']:.3f}, "
            f"required={required_safety_acc:.3f}."
        )

    final_metrics_t1 = evaluate_policy(eval_env_t1, actor, num_episodes=eval_episodes)
    final_metrics_t2 = evaluate_policy(eval_env_t2, actor, num_episodes=eval_episodes)
    _print_metrics("Task 1 final metrics:", final_metrics_t1)
    _print_metrics("Task 2 final metrics:", final_metrics_t2)

    # Save a source-policy trajectory snapshot (Task 1)
    vis_env = make_poisoned_apple_env(
        cfg,
        task_id=1,
        render_mode=None,
        append_task_id=append_task_id,
    )
    try:
        visualize_agent_trajectory(
            env=vis_env,
            actor=actor,
            num_episodes=1,
            env_name="Task_1",
            cfg_name=args.cfg,
            actor_name="source_policy",
            save_dir=str(plots_dir),
        )
    except ModuleNotFoundError as exc:
        print(f"  Skipping trajectory plot (missing optional dependency): {exc}")
    finally:
        vis_env.close()

    # Persist artifacts
    torch.save(actor.state_dict(), out_dir / "source_policy.pt")
    torch.save(critic.state_dict(), out_dir / "source_critic.pt")
    torch.save(training_data, out_dir / "source_training_data.pt")

    if safety_dataset is not None:
        X_s, Y_s = safety_dataset.tensors
        torch.save(
            {
                "X": X_s,
                "Y_multi_hot": Y_s,
                "num_states": int(X_s.shape[0]),
                "num_actions": int(Y_s.shape[1]) if Y_s.ndim == 2 else N_ACTIONS,
            },
            out_dir / "task1_safety_dataset.pt",
        )

    summary = {
        "cfg": args.cfg,
        "seed": seed,
        "ppo_total_steps": total_steps,
        "eval_episodes": eval_episodes,
        "append_task_id": append_task_id,
        "source_metrics_task1": source_metrics_t1,
        "source_metrics_task2": source_metrics_t2,
        "final_metrics_task1": final_metrics_t1,
        "final_metrics_task2": final_metrics_t2,
        "global_safety": global_safety,
        "finetune": finetune_result,
        "safety_dataset_size": (len(safety_dataset) if safety_dataset is not None else 0),
    }

    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "used_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({args.cfg: cfg}, f, sort_keys=False)

    eval_env_t1.close()
    eval_env_t2.close()

    print("\nSaved artifacts:")
    print(f"  - {out_dir / 'source_policy.pt'}")
    print(f"  - {out_dir / 'source_critic.pt'}")
    print(f"  - {out_dir / 'source_training_data.pt'}")
    if safety_dataset is not None:
        print(f"  - {out_dir / 'task1_safety_dataset.pt'}")
    print(f"  - {out_dir / 'training_summary.json'}")
    print(f"  - {out_dir / 'used_config.yaml'}")


if __name__ == "__main__":
    main()
