#!/usr/bin/env python3
"""Train a source policy for Taxi Task 1 using PPO, then optional safety fine-tuning."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from torch.utils.data import TensorDataset
import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # taxi/
_EXP_DIR = _SCRIPT_DIR.parent                          # experiments/
_PROJECT_ROOT = _EXP_DIR.parent.parent                 # CertifiedContinualLearning/
_RL_DIR = _PROJECT_ROOT / "rl_project"
for p in (_EXP_DIR, _RL_DIR, _PROJECT_ROOT, _SCRIPT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from taxi_utils import (
    LOC_NAMES,
    create_taxi_safety_rashomon_dataset,
    evaluate_policy,
    make_taxi_env,
    verify_safety_accuracy,
)
from rl_project.utils.ppo_utils import PPOConfig, ppo_train
from rl_project.utils.gymnasium_utils import plot_episode

# ── constants ───────────────────────────────────────────────────────────────
N_ACTIONS = 6


# ── helpers ─────────────────────────────────────────────────────────────────
def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_actor(obs_dim: int, hidden: int = 128) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, N_ACTIONS),
    )


def _make_critic(obs_dim: int, hidden: int = 128) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )


def _format_loc(loc: int) -> str:
    return LOC_NAMES[loc]


def _early_stop_criteria_met(metrics: dict[str, float], eps: float = 1e-12) -> bool:
    """All eval episodes must be safe and successful."""
    return (
        metrics["avg_safety_success"] >= (1.0 - eps)
        and metrics["avg_success"] >= (1.0 - eps)
    )


def _append_training_data(
    merged: dict[str, np.ndarray] | None,
    chunk: dict,
) -> dict[str, np.ndarray]:
    keys = ("states", "actions", "terminated", "truncated", "safe")
    if merged is None:
        return {k: np.asarray(chunk[k]) for k in keys}
    for k in keys:
        merged[k] = np.concatenate([merged[k], np.asarray(chunk[k])], axis=0)
    return merged


def train_ppo(
    task1_passenger_loc: int,
    task1_dest_loc: int,
    seed: int,
    total_steps: int,
    is_rainy: bool,
    fickle_passenger: bool,
    hidden: int,
    ent_coef: float,
    eval_episodes: int,
    early_stop_check_interval: int,
    device: str,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential, dict[str, np.ndarray], dict[str, float], int]:
    env = make_taxi_env(
        task_num=0,
        passenger_loc=task1_passenger_loc,
        dest_loc=task1_dest_loc,
        is_rainy=is_rainy,
        fickle_passenger=fickle_passenger,
    )
    obs_dim = int(env.observation_space.shape[0])  # type: ignore[index]
    actor = _make_actor(obs_dim, hidden)
    critic = _make_critic(obs_dim, hidden)

    env.close()

    # Train in chunks so we can enforce the exact early-stop criterion:
    #   1) no unsafe actions in any eval episode
    #   2) all eval episodes terminate successfully (no truncation at max length)
    chunk_steps = max(1, min(early_stop_check_interval, total_steps))
    steps_trained = 0
    chunk_idx = 0
    merged_training_data: dict[str, np.ndarray] | None = None
    latest_metrics: dict[str, float] = {
        "avg_reward": float("nan"),
        "avg_success": 0.0,
        "avg_safety_success": 0.0,
        "avg_steps": float("nan"),
    }

    while steps_trained < total_steps:
        this_chunk_steps = min(chunk_steps, total_steps - steps_trained)
        print(
            f"  PPO chunk {chunk_idx + 1}: "
            f"training {this_chunk_steps} steps "
            f"(trained {steps_trained}/{total_steps})"
        )

        train_env = make_taxi_env(
            task_num=0,
            passenger_loc=task1_passenger_loc,
            dest_loc=task1_dest_loc,
            is_rainy=is_rainy,
            fickle_passenger=fickle_passenger,
        )
        cfg = PPOConfig(
            seed=seed + chunk_idx,
            total_timesteps=this_chunk_steps,
            eval_episodes=eval_episodes,
            ent_coef=ent_coef,
            device=device,
            minibatch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            vf_coef=0.5,
            lr=3e-4,
            max_grad_norm=0.5,
        )
        actor, critic, chunk_training_data = ppo_train(
            env=train_env,
            cfg=cfg,
            actor_warm_start=actor,
            critic_warm_start=critic,
            return_training_data=True,
        )
        merged_training_data = _append_training_data(merged_training_data, chunk_training_data)
        steps_trained += this_chunk_steps
        chunk_idx += 1

        eval_env = make_taxi_env(
            task_num=0,
            passenger_loc=task1_passenger_loc,
            dest_loc=task1_dest_loc,
            is_rainy=is_rainy,
            fickle_passenger=fickle_passenger,
        )
        latest_metrics = evaluate_policy(eval_env, actor.cpu(), num_episodes=eval_episodes)
        eval_env.close()
        print(
            "  Early-stop check: "
            f"success={latest_metrics['avg_success']:.3f}, "
            f"safety_success={latest_metrics['avg_safety_success']:.3f}, "
            f"avg_steps={latest_metrics['avg_steps']:.2f}"
        )
        if _early_stop_criteria_met(latest_metrics):
            print(
                "  Early-stop criterion met: all evaluation episodes were safe and "
                "ended successfully within the episode length limit."
            )
            break

    if merged_training_data is None:
        raise RuntimeError("PPO produced no training data.")

    return actor.cpu(), critic.cpu(), merged_training_data, latest_metrics, steps_trained


def finetune_policy(
    policy: torch.nn.Sequential,
    dataset: torch.utils.data.TensorDataset,
    env: gymnasium.Env,
    overlap_mode: str,
    required_accuracy: float,
    lr: float = 1e-2,
    max_epochs: int = 3000,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, object]:
    """Finetune Taxi policy via supervised multi-label classification.

    Training data:
    1) States on the deterministic optimal trajectory with one-hot target
       equal to the policy's trajectory action.
    2) Safety-critical states outside that trajectory with multi-hot target
       equal to the safe actions from the Rashomon safety dataset.
    """
    if required_accuracy > 1.0:
        required_accuracy = required_accuracy / 100.0
    required_accuracy = float(required_accuracy)

    if overlap_mode not in ("safety", "policy"):
        raise ValueError(f"overlap_mode must be 'safety' or 'policy', got '{overlap_mode}'")
    _ = overlap_mode  # kept for backward compatibility; supervision logic is fixed.

    if not hasattr(dataset, "tensors") or len(dataset.tensors) < 2:
        raise ValueError("Expected a TensorDataset-like object with tensors (X, Y).")

    X, Y = dataset.tensors
    if Y.ndim != 2:
        raise ValueError(
            f"Expected multi-label safety targets of shape (N, A). Got shape={tuple(Y.shape)}."
        )

    device = next(policy.parameters()).device
    X = X.to(device)
    Y = Y.to(device)
    torch.manual_seed(seed)

    policy.eval()
    with torch.no_grad():
        n_actions = int(policy(X[:1]).shape[-1])

    # --- Step 1: collect safety labels from Rashomon dataset ---
    safety_state_to_allowed: dict[tuple[float, ...], set[int]] = {}
    for i in range(X.shape[0]):
        key = tuple(float(v) for v in X[i].detach().cpu().tolist())
        valid_actions = torch.where(Y[i] > 0)[0].tolist()
        if len(valid_actions) == 0:
            raise ValueError(f"Safety label at index {i} has no valid actions.")
        if key not in safety_state_to_allowed:
            safety_state_to_allowed[key] = set()
        safety_state_to_allowed[key].update(int(a) for a in valid_actions)

    # --- Step 2: rollout deterministic trajectory and record target actions ---
    max_steps = getattr(getattr(env, "spec", None), "max_episode_steps", 200)
    trajectory_state_to_action: dict[tuple[float, ...], int] = {}

    with torch.no_grad():
        obs, _ = env.reset(seed=seed)
        done = False
        n_steps = 0
        while not done and n_steps < max_steps:
            obs_arr = np.asarray(obs, dtype=np.float32)
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(policy(obs_t).argmax(dim=1).item())
            key = tuple(float(v) for v in obs_arr.tolist())
            if key not in trajectory_state_to_action:
                trajectory_state_to_action[key] = action

            obs, _, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            n_steps += 1

    trajectory_keys = set(trajectory_state_to_action.keys())

    # --- Step 3: build supervised dataset ---
    # Trajectory states -> one-hot optimal action labels.
    combined_state_to_label: dict[tuple[float, ...], np.ndarray] = {}
    for key, action in trajectory_state_to_action.items():
        y = np.zeros(n_actions, dtype=np.float32)
        y[action] = 1.0
        combined_state_to_label[key] = y

    # Off-trajectory safety states -> multi-hot safe-action labels.
    for key, allowed_actions in safety_state_to_allowed.items():
        if key in trajectory_keys:
            continue
        y = np.zeros(n_actions, dtype=np.float32)
        for a in allowed_actions:
            y[a] = 1.0
        combined_state_to_label[key] = y

    if len(combined_state_to_label) == 0:
        raise RuntimeError("No finetuning data was built.")

    keys = list(combined_state_to_label.keys())
    combined_states = torch.tensor(keys, dtype=torch.float32, device=device)
    combined_labels = torch.tensor(
        np.asarray([combined_state_to_label[k] for k in keys]),
        dtype=torch.float32,
        device=device,
    )
    is_trajectory = torch.tensor(
        [k in trajectory_keys for k in keys],
        dtype=torch.bool,
        device=device,
    )
    off_trajectory_mask = ~is_trajectory

    n_total = int(combined_states.shape[0])
    n_off_trajectory = int(off_trajectory_mask.sum().item())
    n_trajectory = int(is_trajectory.sum().item())
    off_trajectory_target = 1.0
    eps = 1e-12

    def _subset_accuracy(preds: torch.Tensor, mask: torch.Tensor) -> float:
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return 1.0
        return float(combined_labels[idx, preds[idx]].mean().item())

    def _all_metrics(logits: torch.Tensor) -> tuple[float, float, float]:
        preds = logits.argmax(dim=1)
        all_idx = torch.arange(n_total, device=device)
        combined_acc = float(combined_labels[all_idx, preds].mean().item())
        traj_acc = _subset_accuracy(preds, is_trajectory)
        off_traj_acc = _subset_accuracy(preds, off_trajectory_mask)
        return combined_acc, traj_acc, off_traj_acc

    if verbose:
        print("\n--- Finetuning policy ---")
        print(f"  Safety dataset states: {X.shape[0]}")
        print(f"  Deterministic trajectory states: {n_trajectory}")
        print(f"  Off-trajectory constrained states: {n_off_trajectory}")
        print(f"  Combined supervised states: {n_total}")

    with torch.no_grad():
        logits0 = policy(combined_states)
        init_acc, init_traj_acc, init_off_trajectory_acc = _all_metrics(logits0)

    if verbose:
        print(
            "  Initial | "
            f"combined_acc={init_acc:.3f}, "
            f"traj_action_acc={init_traj_acc:.3f} (target={required_accuracy:.3f}), "
            f"off_traj_safety={init_off_trajectory_acc:.3f} (target=1.000)"
        )

    if (
        init_traj_acc >= required_accuracy
        and init_off_trajectory_acc >= (off_trajectory_target - eps)
    ):
        if verbose:
            print("  Already satisfies target constraints.")
        return {
            "policy": policy,
            "final_accuracy": init_acc,
            "final_trajectory_accuracy": init_traj_acc,
            "final_off_trajectory_safety_accuracy": init_off_trajectory_acc,
            "target_accuracy": required_accuracy,
            "epochs_run": 0,
            "reached_target": True,
            "combined_dataset": TensorDataset(combined_states, combined_labels),
        }

    # --- Step 4: finetune ---
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    reached = False
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        logits = policy(combined_states)
        target_dist = combined_labels / combined_labels.sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(target_dist * log_probs).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            combined_acc_v, traj_acc_v, off_trajectory_acc_v = _all_metrics(policy(combined_states))

        if verbose and (
            epoch % 100 == 0
            or (
                traj_acc_v >= required_accuracy
                and off_trajectory_acc_v >= (off_trajectory_target - eps)
            )
        ):
            print(
                f"  Epoch {epoch:4d} | loss={loss.item():.6f} | "
                f"combined_acc={combined_acc_v:.3f} | "
                f"traj_action_acc={traj_acc_v:.3f} | "
                f"off_traj_safety={off_trajectory_acc_v:.3f}"
            )

        if traj_acc_v >= required_accuracy and off_trajectory_acc_v >= (off_trajectory_target - eps):
            reached = True
            break

    policy.eval()
    with torch.no_grad():
        final_acc, final_traj_acc, final_off_trajectory_acc = _all_metrics(policy(combined_states))

    if verbose:
        print(
            "  Final | "
            f"combined_acc={final_acc:.3f}, "
            f"traj_action_acc={final_traj_acc:.3f} (target={required_accuracy:.3f}), "
            f"off_traj_safety={final_off_trajectory_acc:.3f} (target=1.000), "
            f"reached={reached}"
        )
        print("--- Finetuning complete ---\n")

    if final_off_trajectory_acc < (off_trajectory_target - eps):
        raise RuntimeError(
            "Off-trajectory safety constraint not satisfied. "
            f"Final off-trajectory safety accuracy={final_off_trajectory_acc:.4f}."
        )

    if final_traj_acc < required_accuracy:
        raise RuntimeError(
            "Trajectory action-preservation constraint not satisfied. "
            f"Final trajectory accuracy={final_traj_acc:.4f}, "
            f"required={required_accuracy:.4f}."
        )

    if not reached:
        raise RuntimeError(
            "Could not satisfy constraints within max_epochs. "
            "Try larger max_epochs or lower lr."
        )

    return {
        "policy": policy,
        "final_accuracy": final_acc,
        "final_trajectory_accuracy": final_traj_acc,
        "final_off_trajectory_safety_accuracy": final_off_trajectory_acc,
        "target_accuracy": required_accuracy,
        "epochs_run": epoch,
        "reached_target": reached,
        "combined_dataset": TensorDataset(combined_states, combined_labels),
    }

# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a source policy for Taxi Task 1.")
    parser.add_argument("--seed", type=int, default=0, help="Override seed (default: from cfg or 0).")
    parser.add_argument("--cfg", type=str, default="different_dest", help="Config key in configs.yaml.")
    parser.add_argument(
        "--safety-finetuning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to do the safety fine-tuning step.",
    )
    parser.add_argument("--total-steps", type=int, default=None, help="Max PPO timesteps.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size for actor and critic.")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=1, help="Number of episodes for PPO evaluation and early-stop checks.")
    parser.add_argument(
        "--early-stop-check-interval",
        type=int,
        default=50_000,
        help="PPO steps between exact early-stop checks.",
    )
    parser.add_argument("--safety-epochs", type=int, default=2_000)
    parser.add_argument("--min-safety-accuracy", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save outputs (default: outputs/<cfg>/<seed>/source).",
    )
    args = parser.parse_args()

    # ── load config ──
    with (_SCRIPT_DIR / "configs.yaml").open("r", encoding="utf-8") as f:
        all_cfgs = yaml.safe_load(f)
    if args.cfg not in all_cfgs:
        raise ValueError(f"Config '{args.cfg}' not in configs.yaml. Available: {list(all_cfgs)}")
    cfg = all_cfgs[args.cfg]

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    total_steps = int(
        args.total_steps
        if args.total_steps is not None
        else cfg.get("source_total_timesteps", cfg.get("noadapt_train_timesteps", 500_000))
    )
    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else cfg.get("eval_episodes", 100))
    eval_episodes = max(1, eval_episodes)

    task1_passenger_loc: int = int(cfg["task1_passenger_loc"])
    task1_dest_loc: int = int(cfg["task1_dest_loc"])
    is_rainy: bool = bool(cfg.get("is_rainy", False))
    fickle_passenger: bool = bool(cfg.get("fickle_passenger", False))

    out_dir = Path(args.output_dir) if args.output_dir else _SCRIPT_DIR / "outputs" / args.cfg / str(seed) / "source"
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seeds(seed)

    # ── Step 1: Train PPO on Task 1 ──
    print("=" * 60)
    print(f"Step 1 — PPO training  (seed={seed}, cfg={args.cfg})")
    print("=" * 60)
    print(f"  Task 1 routes: pickup [{_format_loc(task1_passenger_loc)}] -> deliver [{_format_loc(task1_dest_loc)}]")
    print(f"  Deterministic dynamics: {not is_rainy and not fickle_passenger}")

    actor, critic, training_data, early_stop_metrics, steps_trained = train_ppo(
        task1_passenger_loc=task1_passenger_loc,
        task1_dest_loc=task1_dest_loc,
        seed=seed,
        total_steps=max(1, total_steps),
        is_rainy=is_rainy,
        fickle_passenger=fickle_passenger,
        hidden=args.hidden,
        ent_coef=args.ent_coef,
        eval_episodes=eval_episodes,
        early_stop_check_interval=max(1, args.early_stop_check_interval),
        device=args.device,
    )
    print(f"  PPO trained steps (effective): {steps_trained}/{total_steps}")
    print(
        "  Last early-stop check: "
        f"success={early_stop_metrics['avg_success']:.3f}, "
        f"safety_success={early_stop_metrics['avg_safety_success']:.3f}"
    )

    # Quick verification
    env = make_taxi_env(
        task_num=0,
        passenger_loc=task1_passenger_loc,
        dest_loc=task1_dest_loc,
        is_rainy=is_rainy,
        fickle_passenger=fickle_passenger,
    )
    obs, _ = env.reset(seed=args.seed)
    done, total_reward = False, 0.0
    while not done:
        with torch.no_grad():
            action = int(torch.argmax(actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)), 1).item())
        obs, r, term, trunc, _ = env.step(action)
        total_reward += r # type: ignore
        done = term or trunc
    env.close()
    print(f"  PPO deterministic total reward = {total_reward:.1f}")
    if total_reward < 1.0:
        raise RuntimeError("PPO did not learn the optimal policy (reward < 1.0).")
    
    # ── Step 2: Fine-tune for safety (OPTIONAL) ──
    if args.safety_finetuning:
        print("\n" + "=" * 60)
        print("Step 2 — Safety fine-tuning")
        print("=" * 60)
        print("  Finetuning source policy for safety …")
        safety_rashomon_dataset = create_taxi_safety_rashomon_dataset(
            task_num=0,
            passenger_loc=task1_passenger_loc,
            dest_loc=task1_dest_loc,
        )
        finetuning_result_dct = finetune_policy(
            policy=actor,
            dataset=safety_rashomon_dataset,
            env=make_taxi_env(
                task_num=0,
                passenger_loc=task1_passenger_loc,
                dest_loc=task1_dest_loc,
                is_rainy=is_rainy,
                fickle_passenger=fickle_passenger,
            ),
            overlap_mode="policy",
            required_accuracy=1.0,
        )
        actor = finetuning_result_dct['policy']
        if not finetuning_result_dct['reached_target']:
            raise ValueError(
                f"Safety finetuning did not reach required accuracy. "
                f"Final accuracy: {finetuning_result_dct['final_accuracy']:.4f}, "
                f"required: {finetuning_result_dct['target_accuracy']:.4f}"
            )

        # Additional check
        safety_ref_acc, _ = verify_safety_accuracy(
            actor=actor,
            dataset=safety_rashomon_dataset,
            multi_label=True,
            min_acc_limit=args.min_safety_accuracy,
            verbose=False,
        )
        if safety_ref_acc < 1.0:
            raise RuntimeError(
                "Post-finetuning safety verification failed: "
                f"{safety_ref_acc:.4f} < 1.0000"
            )

    # Plot the learned policy
    plot_episode(
        env=make_taxi_env(
            task_num=0,
            passenger_loc=task1_passenger_loc,
            dest_loc=task1_dest_loc,
            is_rainy=is_rainy,
            fickle_passenger=fickle_passenger,
            render_mode="rgb_array"
        ),
        actor=actor.cpu(),
        seed=args.seed,
        save_path=str(out_dir / "source_policy_trajectory.png"),
        figsize_per_frame=(3.0, 3.0),
        title='Source Policy Trajectory (Task 1)'
    )

    # ── Step 3: Save model ──
    print("\n" + "=" * 60)
    print("Step 3 — Saving neural policy")
    print("=" * 60)

    model_path = out_dir / "source_policy.pt"
    torch.save(actor.state_dict(), model_path)
    print(f"  Saved model → {model_path}")

    critic_path = out_dir / "source_critic.pt"
    torch.save(critic.state_dict(), critic_path)
    print(f"  Saved critic → {critic_path}")

    training_data_path = out_dir / "source_training_data.pt"
    torch.save(training_data, training_data_path)
    print(f"  Saved training data → {training_data_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
