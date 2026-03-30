#!/usr/bin/env python3
"""
Downstream adaptation experiment for Taxi.

Steps:
  1. Load the source policy, critic, and training data.
  2. Build a safety Rashomon dataset from Task 1 and compute safety bounds.
  3. SafeAdapt   — adapt to Task 2 using safety Rashomon bounds.
  4. UnsafeAdapt — adapt to Task 2 without any bounds.
  5. EWC         — adapt to Task 2 using Elastic Weight Consolidation.
  6. Evaluate all policies in Task 1 and Task 2; save a summary table.
  7. Save model checkpoints, plots, and a JSON summary.

Notes:
  - PerfAdapt is intentionally not implemented for Taxi in this script.
  - Downstream early stop uses the Taxi source-policy criterion:
      all evaluation episodes must be safe and successful.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXP_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _EXP_DIR.parent.parent
_RL_DIR = _PROJECT_ROOT / "rl_project"
for p in (_SCRIPT_DIR, _EXP_DIR, _PROJECT_ROOT, _RL_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from taxi_utils import (
    LOC_NAMES,
    create_taxi_safety_rashomon_dataset,
    evaluate_policy,
    make_taxi_env,
    train_safety_actor,
    verify_safety_accuracy,
)
from rl_project.utils.gymnasium_utils import plot_episode
from rl_project.utils.ppo_utils import PPOConfig, ppo_train
from rl_project.utils.ewc_ppo import EWCPPOConfig, compute_ewc_state, ewc_ppo_train
from src.trainer.IntervalTrainer import IntervalTrainer


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream adaptation experiment")
    p.add_argument("--cfg", default="different_dest", help="Config key in configs.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help=(
            "Directory with source_policy.pt / source_critic.pt / source_training_data.pt, "
            "or run directory containing source/. "
            "Defaults to outputs/<cfg>/<seed>."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save results. Defaults to <run_dir>/downstream",
    )
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Max total timesteps for downstream adaptation",
    )
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ewc-lambda", type=float, default=5_000.0)
    p.add_argument("--rashomon-n-iters", type=int, default=20_000) # used to be 5_000
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--hidden", type=int, default=128, help="Hidden layer size for actor and critic networks")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--source-mode",
        type=str,
        default="safe",
        choices=["original", "safe"],
        help=(
            "'original': use source policy as-is for safety-based adaptation; "
            "'safe': first safety-refine source policy before safety-based adaptation"
        ),
    )
    p.add_argument(
        "--min-safety-accuracy",
        type=float,
        default=1.0,
        help="Required Task-1 safety accuracy used for certification and checks",
    )
    p.add_argument("--safety-epochs", type=int, default=2_000)
    p.add_argument(
        "--early-stop-check-interval",
        type=int,
        default=50_000,
        help="PPO steps between downstream early-stop checks.",
    )
    return p.parse_args()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_actor(obs_dim: int, n_actions: int, hidden: int = 64) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, n_actions),
    )


def _make_critic(obs_dim: int, hidden: int = 64) -> torch.nn.Sequential:
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
    """Taxi criterion: every eval episode is safe and successful."""
    return (
        metrics["avg_safety_success"] >= (1.0 - eps)
        and metrics["avg_success"] >= (1.0 - eps)
    )


def compute_rashomon_bounds(
    *,
    actor: torch.nn.Module,
    rashomon_dataset,
    seed: int,
    rashomon_n_iters: int,
    dataset_name: str,
    min_hard_specification: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, float, int, float, int]:
    """Compute Rashomon bounds for a multi-label dataset (FrozenLake-style logic)."""
    n_safe_per_state = rashomon_dataset.tensors[1].sum(dim=1).tolist()
    max_safe_actions_per_state = max(n_safe_per_state) if n_safe_per_state else 0
    min_surrogate_threshold = max_safe_actions_per_state / (1 + max_safe_actions_per_state)

    print(f"  [{dataset_name}] Rashomon states: {len(n_safe_per_state)}")
    print(f"  [{dataset_name}] Surrogate threshold: {min_surrogate_threshold:.6f}")

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
            raise ValueError(
                f"[{dataset_name}] Could not satisfy surrogate threshold with inverse_temp <= 1000. "
                f"Worst state idx={worst_idx}, mass={safe_prob_mass[worst_idx].item():.6f}, "  # type: ignore[index]
                f"required={min_surrogate_threshold:.6f}."
            )

    print(f"  [{dataset_name}] Smallest inverse_temp: {inverse_temp}")
    print(
        f"  [{dataset_name}] Min safe-action softmax mass at T={inverse_temp}: "
        f"{safe_prob_mass.min().item():.6f}"  # type: ignore[union-attr]
    )

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
        best = max(interval_trainer.certificates)
        raise ValueError(
            f"[{dataset_name}] No Rashomon certificate satisfies hard specification "
            f"({min_hard_specification:.4f}). Best={best:.4f}"
        )

    final_idx = valid_idxs[-1]
    cert = float(interval_trainer.certificates[final_idx])
    print(f"  [{dataset_name}] Certified hard specification: {cert:.4f}")

    bounded_model = interval_trainer.bounds[final_idx]
    param_bounds_l = [p.detach().cpu() for p in bounded_model.param_l]
    param_bounds_u = [p.detach().cpu() for p in bounded_model.param_u]

    return (
        param_bounds_l,
        param_bounds_u,
        bounded_model,
        cert,
        final_idx,
        float(min_surrogate_threshold),
        int(inverse_temp),
    )


def train_ppo_with_taxi_early_stop(
    *,
    make_env,
    actor_init: torch.nn.Module,
    critic_init: torch.nn.Module,
    seed: int,
    total_timesteps: int,
    ent_coef: float,
    eval_episodes: int,
    early_stop_check_interval: int,
    device: str,
    actor_param_bounds_l: list[torch.Tensor] | None = None,
    actor_param_bounds_u: list[torch.Tensor] | None = None,
    method_name: str,
) -> tuple[torch.nn.Module, torch.nn.Module, int, dict[str, float]]:
    """Train Task-2 PPO in chunks with Taxi exact early-stop criterion."""
    actor = copy.deepcopy(actor_init).cpu()
    critic = copy.deepcopy(critic_init).cpu()

    check_interval = max(1, min(early_stop_check_interval, total_timesteps))
    steps_trained = 0
    chunk_idx = 0
    latest_metrics: dict[str, float] = {
        "avg_reward": float("nan"),
        "avg_success": 0.0,
        "avg_safety_success": 0.0,
        "avg_steps": float("nan"),
    }

    while steps_trained < total_timesteps:
        this_chunk = min(check_interval, total_timesteps - steps_trained)
        print(
            f"  [{method_name}] chunk {chunk_idx + 1}: "
            f"training {this_chunk} steps (trained {steps_trained}/{total_timesteps})"
        )

        env2_train = make_env(1)
        cfg = PPOConfig(
            seed=seed + chunk_idx,
            total_timesteps=this_chunk,
            ent_coef=ent_coef,
            eval_episodes=eval_episodes,
            device=device,
            early_stop=False,
        )
        actor, critic = ppo_train(
            env=env2_train,
            cfg=cfg,
            actor_warm_start=actor,  # type: ignore[arg-type]
            critic_warm_start=critic,  # type: ignore[arg-type]
            actor_param_bounds_l=actor_param_bounds_l,
            actor_param_bounds_u=actor_param_bounds_u,
        )

        steps_trained += this_chunk
        chunk_idx += 1

        env2_eval = make_env(1)
        latest_metrics = evaluate_policy(env2_eval, actor.cpu(), num_episodes=eval_episodes)
        env2_eval.close()
        print(
            f"  [{method_name}] early-stop check: "
            f"success={latest_metrics['avg_success']:.3f}, "
            f"safety_success={latest_metrics['avg_safety_success']:.3f}, "
            f"avg_steps={latest_metrics['avg_steps']:.2f}"
        )

        if _early_stop_criteria_met(latest_metrics):
            print(
                f"  [{method_name}] early-stop criterion met: all eval episodes "
                "were safe and successful."
            )
            break

    return actor.cpu(), critic.cpu(), steps_trained, latest_metrics


def train_ewc_with_taxi_early_stop(
    *,
    make_env,
    actor_init: torch.nn.Module,
    critic_init: torch.nn.Module,
    ewc_states: list,
    seed: int,
    total_timesteps: int,
    ent_coef: float,
    eval_episodes: int,
    early_stop_check_interval: int,
    ewc_lambda: float,
    device: str,
    method_name: str,
) -> tuple[torch.nn.Module, torch.nn.Module, int, dict[str, float]]:
    """Train Task-2 EWC PPO in chunks with Taxi exact early-stop criterion."""
    actor = copy.deepcopy(actor_init).cpu()
    critic = copy.deepcopy(critic_init).cpu()

    check_interval = max(1, min(early_stop_check_interval, total_timesteps))
    steps_trained = 0
    chunk_idx = 0
    latest_metrics: dict[str, float] = {
        "avg_reward": float("nan"),
        "avg_success": 0.0,
        "avg_safety_success": 0.0,
        "avg_steps": float("nan"),
    }

    while steps_trained < total_timesteps:
        this_chunk = min(check_interval, total_timesteps - steps_trained)
        print(
            f"  [{method_name}] chunk {chunk_idx + 1}: "
            f"training {this_chunk} steps (trained {steps_trained}/{total_timesteps})"
        )

        env2_train = make_env(1)
        cfg = EWCPPOConfig(
            seed=seed + chunk_idx,
            total_timesteps=this_chunk,
            ent_coef=ent_coef,
            eval_episodes=eval_episodes,
            ewc_lambda=ewc_lambda,
            device=device,
            early_stop=False,
        )
        actor, critic = ewc_ppo_train(
            env=env2_train,
            cfg=cfg,
            ewc_states=ewc_states,
            actor_warm_start=actor,  # type: ignore[arg-type]
            critic_warm_start=critic,  # type: ignore[arg-type]
        )

        steps_trained += this_chunk
        chunk_idx += 1

        env2_eval = make_env(1)
        latest_metrics = evaluate_policy(env2_eval, actor.cpu(), num_episodes=eval_episodes)
        env2_eval.close()
        print(
            f"  [{method_name}] early-stop check: "
            f"success={latest_metrics['avg_success']:.3f}, "
            f"safety_success={latest_metrics['avg_safety_success']:.3f}, "
            f"avg_steps={latest_metrics['avg_steps']:.2f}"
        )

        if _early_stop_criteria_met(latest_metrics):
            print(
                f"  [{method_name}] early-stop criterion met: all eval episodes "
                "were safe and successful."
            )
            break

    return actor.cpu(), critic.cpu(), steps_trained, latest_metrics


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Load config ─────────────────────────────────────────────────────
    config_path = _SCRIPT_DIR / "configs.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        all_cfgs = yaml.safe_load(f)
    if args.cfg not in all_cfgs:
        raise ValueError(f"Config '{args.cfg}' not found. Available: {list(all_cfgs)}")

    cfg = all_cfgs[args.cfg]
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    total_timesteps = int(
        args.total_timesteps
        if args.total_timesteps is not None
        else cfg.get("downstream_total_timesteps", cfg.get("safeadapt_train_timesteps", 200_000))
    )
    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else cfg.get("eval_episodes", 100))
    eval_episodes = max(1, eval_episodes)

    task1_passenger_loc: int = int(cfg["task1_passenger_loc"])
    task1_dest_loc: int = int(cfg["task1_dest_loc"])
    task2_passenger_loc: int = int(cfg["task2_passenger_loc"])
    task2_dest_loc: int = int(cfg["task2_dest_loc"])
    is_rainy: bool = bool(cfg.get("is_rainy", False))
    fickle_passenger: bool = bool(cfg.get("fickle_passenger", False))

    def make_env(task: int, render_mode: str | None = None):
        return make_taxi_env(
            task_num=task,
            passenger_loc=task1_passenger_loc if task == 0 else task2_passenger_loc,
            dest_loc=task1_dest_loc if task == 0 else task2_dest_loc,
            is_rainy=is_rainy,
            fickle_passenger=fickle_passenger,
            render_mode=render_mode,
        )

    _set_seeds(seed)

    env_tmp = make_env(task=0)
    obs_dim = int(env_tmp.observation_space.shape[0])  # type: ignore[index]
    n_actions = int(env_tmp.action_space.n)  # type: ignore[union-attr]
    env_tmp.close()

    # ── Paths ───────────────────────────────────────────────────────────
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
    print("DOWNSTREAM ADAPTATION EXPERIMENT (Taxi)")
    print("=" * 80)
    print(f"Config         : {args.cfg}")
    print(f"Seed           : {seed}")
    print(
        f"Task 1 routes  : pickup [{_format_loc(task1_passenger_loc)}] "
        f"-> deliver [{_format_loc(task1_dest_loc)}]"
    )
    print(
        f"Task 2 routes  : pickup [{_format_loc(task2_passenger_loc)}] "
        f"-> deliver [{_format_loc(task2_dest_loc)}]"
    )
    print(f"Deterministic  : {not is_rainy and not fickle_passenger}")
    print(f"Source dir     : {source_dir}")
    print(f"Run dir        : {run_dir}")
    print(f"Downstream dir : {downstream_dir}")
    print(f"Timesteps      : {total_timesteps}")
    print(f"Eval episodes  : {eval_episodes}")
    print("=" * 80)

    # ── 1. Load source policy ───────────────────────────────────────────
    print("\n[1/7] Loading source policy, critic, and training data …")
    source_actor = _make_actor(obs_dim, n_actions, hidden=args.hidden)
    source_actor.load_state_dict(torch.load(source_dir / "source_policy.pt", map_location="cpu"))

    source_critic = _make_critic(obs_dim, hidden=args.hidden)
    source_critic.load_state_dict(torch.load(source_dir / "source_critic.pt", map_location="cpu"))

    source_training_data = torch.load(
        source_dir / "source_training_data.pt", map_location="cpu", weights_only=False,
    )
    print(f"  Source training data: {len(source_training_data['states'])} transitions")

    # ── 2. Build safety Rashomon dataset and bounds ─────────────────────
    print("\n[2/7] Building safety Rashomon dataset and bounds …")
    safety_rashomon_dataset = create_taxi_safety_rashomon_dataset(
        task_num=0,
        passenger_loc=task1_passenger_loc,
        dest_loc=task1_dest_loc,
    )
    torch.save(safety_rashomon_dataset, downstream_dir / "rashomon_dataset_safety.pt")
    torch.save(safety_rashomon_dataset, downstream_dir / "rashomon_dataset.pt")  # backward compatible
    print(f"  Safety-critical states: {len(safety_rashomon_dataset)}")

    source_for_safeadapt = copy.deepcopy(source_actor).cpu()
    safety_ref_acc, _ = verify_safety_accuracy(
        actor=source_for_safeadapt,
        dataset=safety_rashomon_dataset,
        multi_label=True,
        min_acc_limit=args.min_safety_accuracy,
        verbose=False,
    )

    (
        safe_param_bounds_l,
        safe_param_bounds_u,
        safe_bounded_model,
        selected_cert,
        selected_bound_index,
        surrogate_threshold,
        inverse_temp,
    ) = compute_rashomon_bounds(
        actor=source_for_safeadapt.cpu(),
        rashomon_dataset=safety_rashomon_dataset,
        seed=seed,
        rashomon_n_iters=args.rashomon_n_iters,
        dataset_name="Safety",
        min_hard_specification=args.min_safety_accuracy,
    )
    torch.save(safe_bounded_model, downstream_dir / "bounded_model_safety.pt")
    torch.save(safe_bounded_model, downstream_dir / "bounded_model.pt")  # backward compatible

    # ── 3. SafeAdapt — PPO with safety Rashomon bounds ──────────────────
    print("\n[3/7] SafeAdapt: PPO with safety Rashomon bounds …")
    _set_seeds(seed)
    safeadapt_actor, safeadapt_critic, safe_steps, safe_last_metrics = train_ppo_with_taxi_early_stop(
        make_env=make_env,
        actor_init=source_for_safeadapt,
        critic_init=source_critic,
        seed=seed,
        total_timesteps=total_timesteps,
        ent_coef=args.ent_coef,
        eval_episodes=eval_episodes,
        early_stop_check_interval=max(1, args.early_stop_check_interval),
        device=args.device,
        actor_param_bounds_l=safe_param_bounds_l,
        actor_param_bounds_u=safe_param_bounds_u,
        method_name="SafeAdapt",
    )

    # ── 4. UnsafeAdapt — PPO without bounds ─────────────────────────────
    print("\n[4/7] UnsafeAdapt: PPO without any bounds …")
    _set_seeds(seed)
    unsafeadapt_actor, unsafeadapt_critic, unsafe_steps, unsafe_last_metrics = train_ppo_with_taxi_early_stop(
        make_env=make_env,
        actor_init=source_actor,
        critic_init=source_critic,
        seed=seed,
        total_timesteps=total_timesteps,
        ent_coef=args.ent_coef,
        eval_episodes=eval_episodes,
        early_stop_check_interval=max(1, args.early_stop_check_interval),
        device=args.device,
        method_name="UnsafeAdapt",
    )

    # ── 5. EWC — PPO with EWC regularisation ────────────────────────────
    print("\n[5/7] EWC: Computing Fisher information and training …")
    _set_seeds(seed)
    ewc_state = compute_ewc_state(
        actor=copy.deepcopy(source_actor),
        observations=source_training_data["states"],
        compute_critic=False,
        device=args.device,
        fisher_sample_size=min(1000, len(source_training_data["states"])),
        seed=seed,
    )

    ewc_actor, ewc_critic, ewc_steps, ewc_last_metrics = train_ewc_with_taxi_early_stop(
        make_env=make_env,
        actor_init=source_actor,
        critic_init=source_critic,
        ewc_states=[ewc_state],
        seed=seed,
        total_timesteps=total_timesteps,
        ent_coef=args.ent_coef,
        eval_episodes=eval_episodes,
        early_stop_check_interval=max(1, args.early_stop_check_interval),
        ewc_lambda=args.ewc_lambda,
        device=args.device,
        method_name="EWC",
    )

    # ── 6. Evaluate all policies ────────────────────────────────────────
    print("\n[6/7] Evaluating all policies …")

    source_actor = source_actor.cpu()
    safeadapt_actor = safeadapt_actor.cpu()
    unsafeadapt_actor = unsafeadapt_actor.cpu()
    ewc_actor = ewc_actor.cpu()

    policies = {
        "Source": source_actor,
        "SafeAdapt": safeadapt_actor,
        "UnsafeAdapt": unsafeadapt_actor,
        "EWC": ewc_actor,
    }

    rows: list[dict] = []
    for name, actor in policies.items():
        actor.eval()
        for task in (0, 1):
            env_eval = make_env(task)
            metrics = evaluate_policy(env_eval, actor, num_episodes=eval_episodes)
            env_eval.close()

            critical_state_safety = None
            if task == 0:
                critical_state_safety, _ = verify_safety_accuracy(
                    actor=actor,
                    dataset=safety_rashomon_dataset,
                    multi_label=True,
                    min_acc_limit=args.min_safety_accuracy,
                    verbose=False,
                )

            rows.append(
                {
                    "Policy": name,
                    "Task": task + 1,
                    "Trajectory Safety Rate": metrics["avg_safety_success"],
                    "Critical State Safety Rate": critical_state_safety,
                    "Avg Total Reward": metrics["avg_reward"],
                    "Success Rate": metrics["avg_success"],
                    "Avg Steps": metrics["avg_steps"],
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

    # ── 7. Save policy networks, summary, and plots ─────────────────────
    print("\n[7/7] Saving policy checkpoints, summary, and plots …")
    torch.save(safeadapt_actor.state_dict(), downstream_dir / "safeadapt_actor.pt")
    torch.save(unsafeadapt_actor.state_dict(), downstream_dir / "unsafeadapt_actor.pt")
    torch.save(ewc_actor.state_dict(), downstream_dir / "ewc_actor.pt")

    summary_payload = {
        "cfg": args.cfg,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "ent_coef": args.ent_coef,
        "ewc_lambda": args.ewc_lambda,
        "rashomon_n_iters": args.rashomon_n_iters,
        "eval_episodes": eval_episodes,
        "device": args.device,
        "min_safety_accuracy": args.min_safety_accuracy,
        "selected_certificate": selected_cert,
        "selected_bound_index": selected_bound_index,
        "surrogate_threshold": surrogate_threshold,
        "inverse_temp": inverse_temp,
        "safety_reference_accuracy": safety_ref_acc,
        "safeadapt_steps_trained": safe_steps,
        "unsafeadapt_steps_trained": unsafe_steps,
        "ewc_steps_trained": ewc_steps,
        "safeadapt_last_early_stop_metrics": safe_last_metrics,
        "unsafeadapt_last_early_stop_metrics": unsafe_last_metrics,
        "ewc_last_early_stop_metrics": ewc_last_metrics,
        "task1_passenger_loc": task1_passenger_loc,
        "task1_dest_loc": task1_dest_loc,
        "task2_passenger_loc": task2_passenger_loc,
        "task2_dest_loc": task2_dest_loc,
        "is_rainy": is_rainy,
        "fickle_passenger": fickle_passenger,
        "perfadapt_implemented": False,
    }
    with (downstream_dir / "downstream_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("  Saving trajectory plots …")
    for name, actor in policies.items():
        actor.eval()
        for task in (0, 1):
            render_env = make_env(task, render_mode="rgb_array")
            save_path = plots_dir / f"{name}_Task{task + 1}_episode.png"
            plot_episode(
                actor=actor,
                env=render_env,
                deterministic=True,
                seed=seed,
                title=f"{name} — Task {task + 1}",
                save_path=str(save_path),
            )
            render_env.close()
            plt.close("all")

    print(f"\nAll downstream artifacts saved to: {downstream_dir}")
    print(f"All plots saved to: {plots_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
