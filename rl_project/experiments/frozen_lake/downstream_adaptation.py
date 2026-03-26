"""
Downstream adaptation experiment for FrozenLake.

Steps:
  1. Load the source policy, critic, and training data.
  2. Build a safety (Rashomon) specification dataset from Task 1.
  3. Compute the Rashomon set with sound safety surrogate.
  4. SafeAdapt  — adapt to Task 2 using Rashomon parameter bounds (PGD).
  5. UnsafeAdapt — adapt to Task 2 without any bounds.
  6. EWC         — adapt to Task 2 using Elastic Weight Consolidation.
  7. Evaluate all policies in Task 1 and Task 2; save trajectory plots and a
     summary table.

Usage:
  python downstream_adaptation.py
  python downstream_adaptation.py --cfg standard_4x4 --seed 42 --output-dir results/my_run
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path
import gymnasium
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import yaml

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Custom utils
from frozenlake_utils import (
    evaluate_policy,
    make_frozenlake_env,
    verify_safety_posthoc,
)
from rl_project.utils.gymnasium_utils import plot_episode, plot_state_action_pairs
from rl_project.utils.ppo_utils import PPOConfig, ppo_train
from rl_project.utils.ewc_ppo import EWCPPOConfig, ewc_ppo_train, compute_ewc_state
from rl_project.experiments_neurips.frozenlake_safety import finetune_policy
from src.trainer.IntervalTrainer import IntervalTrainer

# ── path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
for p in (str(_SCRIPT_DIR), str(_SCRIPT_DIR.parent), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream adaptation experiment")
    p.add_argument("--cfg", default="standard_4x4", help="Config key in configs.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-dir", type=str, default=None,
                   help="Directory with source_policy.pt / source_critic.pt / source_training_data.pt. "
                        "Defaults to outputs/<cfg>/<seed>/source")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to save results. Defaults to outputs/<cfg>/<seed>/downstream")
    p.add_argument(
        "--total-timesteps", type=int, default=50_000, help="Max total timesteps for downstream adaptation"
    )
    p.add_argument("--ent-coef", type=float, default=0.1)
    p.add_argument("--ewc-lambda", type=float, default=5_000.0)
    p.add_argument("--rashomon-n-iters", type=int, default=5_000)
    p.add_argument(
        "--eval-episodes", type=int, default=1, 
        help="Number of eval episodes for policy metrics calculation. " \
        "Default is 1 because FrozenLake is deterministic."
    )
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--source-mode", type=str, default="safe", choices=["original", "safe"],
                   help="'original': use source policy as-is; "
                        "'safe': finetune source to be safe in all critical states while preserving trajectory")
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


# ── Rashomon safety dataset ────────────────────────────────────────────────
def create_frozenlake_safety_rashomon_dataset(env, task_flag: float = 0.0):
    """
    Create a TensorDataset containing only safety-critical states and their safe actions.

    - X: one-hot observations (optionally with final task-flag dimension)
    - Y: multi-hot vectors of length n_actions, with 1s for safe actions
    """
    desc = env.unwrapped.desc
    grid = [
        "".join(ch.decode() if isinstance(ch, (bytes, bytearray)) else str(ch) for ch in row)
        for row in desc
    ]
    nrows, ncols = len(grid), len(grid[0])
    n_states = nrows * ncols
    n_actions = 4  # FrozenLake: Left, Down, Right, Up

    # Infer observation size from env wrapper
    if hasattr(env.observation_space, "shape") and len(env.observation_space.shape) > 0:
        obs_dim_local = int(env.observation_space.shape[0])
    elif hasattr(env.observation_space, "n"):
        obs_dim_local = int(env.observation_space.n)
    else:
        raise ValueError("Cannot infer observation dimension from env.observation_space.")

    if obs_dim_local not in (n_states, n_states + 1):
        raise ValueError(f"Unsupported obs_dim={obs_dim_local}. Expected {n_states} or {n_states + 1}.")

    def state_to_rc(s: int):
        return s // ncols, s % ncols

    def rc_to_state(r: int, c: int):
        return r * ncols + c

    action_deltas = {
        0: (0, -1),  # Left
        1: (1, 0),   # Down
        2: (0, 1),   # Right
        3: (-1, 0),  # Up
    }

    # Identify hole states
    hole_states = set()
    for r in range(nrows):
        for c in range(ncols):
            if grid[r][c] == "H":
                hole_states.add(rc_to_state(r, c))

    obs_list = []
    label_list = []

    for s in range(n_states):
        r, c = state_to_rc(s)
        cell = grid[r][c]

        # Skip terminal/non-traversable states
        if cell in ("H", "G"):
            continue

        safe_actions = []
        for a, (dr, dc) in action_deltas.items():
            nr, nc = r + dr, c + dc
            hits_wall = (nr < 0 or nr >= nrows or nc < 0 or nc >= ncols)

            if hits_wall:
                # In FrozenLake, wall-hit keeps agent in place (safe)
                safe_actions.append(a)
            else:
                ns = rc_to_state(nr, nc)
                if ns not in hole_states:
                    safe_actions.append(a)

        # Keep only safety-critical states (at least one unsafe action exists)
        if len(safe_actions) == n_actions:
            continue

        obs = np.zeros(obs_dim_local, dtype=np.float32)
        obs[s] = 1.0
        if obs_dim_local == n_states + 1:
            obs[-1] = float(task_flag)

        multi_hot = np.zeros(n_actions, dtype=np.float32)
        for a in safe_actions:
            multi_hot[a] = 1.0
        obs_list.append(obs)
        label_list.append(multi_hot)

    if len(obs_list) == 0:
        raise RuntimeError("No safety-critical states found; dataset is empty.")

    obs_tensor = torch.tensor(np.asarray(obs_list), dtype=torch.float32)
    label_tensor = torch.tensor(np.asarray(label_list), dtype=torch.float32)
    return TensorDataset(obs_tensor, label_tensor)

# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    _set_seeds(args.seed)

    # ── Load config ─────────────────────────────────────────────────────
    config_path = _SCRIPT_DIR / "configs.yaml"
    with open(config_path) as f:
        all_cfgs = yaml.safe_load(f)
    cfg = all_cfgs[args.cfg]

    env1_map: list[str] = cfg["env1_map"]
    env2_map: list[str] = cfg["env2_map"]
    is_slippery: bool = bool(cfg.get("is_slippery", False))

    def make_env(task: int, render_mode: str | None = None):
        return make_frozenlake_env(
            env_map=env1_map if task == 0 else env2_map,
            task_num=task,
            is_slippery=is_slippery,
            render_mode=render_mode,
        )

    env_tmp = make_env(0)
    obs_dim: int = env_tmp.observation_space.shape[0] # type: ignore
    n_actions: int = env_tmp.action_space.n # type: ignore
    env_tmp.close()

    # ── Paths ───────────────────────────────────────────────────────────
    out_dir = Path(args.source_dir) if args.source_dir else (
        _SCRIPT_DIR / "outputs" / args.cfg / str(args.seed)
    )
    source_dir = out_dir / "source"
    downstream_dir = out_dir / "downstream"
    downstream_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DOWNSTREAM ADAPTATION EXPERIMENT")
    print("=" * 80)
    print(f"Config         : {args.cfg}")
    print(f"Seed           : {args.seed}")
    print(f"Source dir     : {source_dir}")
    print(f"Output dir     : {out_dir}")
    print(f"Downstream dir : {downstream_dir}")
    print(f"Timesteps      : {args.total_timesteps}")
    print(f"EWC lambda     : {args.ewc_lambda}")
    print(f"Eval episodes  : {args.eval_episodes}")
    print(f"Source mode    : {args.source_mode}")
    print("=" * 80)

    # ── 1. Load source policy ───────────────────────────────────────────
    print("\n[1/6] Loading source policy, critic, and training data …")
    source_actor = _make_actor(obs_dim, n_actions, hidden=args.hidden)
    source_actor.load_state_dict(
        torch.load(source_dir / "source_policy.pt", map_location="cpu")
    )
    source_critic = _make_critic(obs_dim, hidden=args.hidden)
    source_critic.load_state_dict(
        torch.load(source_dir / "source_critic.pt", map_location="cpu")
    )
    source_training_data = torch.load(
        source_dir / "source_training_data.pt", map_location="cpu", weights_only=False,
    )
    print(f"  Source training data: {len(source_training_data['states'])} transitions")

    # ── 2. Build safety specification & Rashomon set ────────────────────
    print("\n[2/6] Building safety dataset and computing Rashomon set …")
    env1_rashomon = make_frozenlake_env(env_map=env1_map, task_num=0, is_slippery=is_slippery)
    rashomon_dataset = create_frozenlake_safety_rashomon_dataset(env1_rashomon, task_flag=0.0)
    # Save Rashomon dataset for reproducibility/analysis
    torch.save(rashomon_dataset, downstream_dir / "rashomon_dataset.pt")

    env1_rashomon.close()
    # Visualize the safety-critical state-action pairs
    safety_state_action_pairs = []
    for obs, label in rashomon_dataset:
        state_idx = obs[:-1].argmax().item()  # get state index from one-hot (ignore task flag)
        safe_actions = torch.where(label > 0)[0].tolist()
        for a in safe_actions:
            safety_state_action_pairs.append((state_idx, a))
    env_plot = make_frozenlake_env(env_map=env1_map, task_num=0, is_slippery=is_slippery, render_mode="rgb_array")
    _ = plot_state_action_pairs(
        env=env_plot,
        state_action_pairs=safety_state_action_pairs,
        arrow_color="teal",
        title="Rashomon state-action pairs (Task 1)",
        save_path=str(plots_dir / "rashomon_state_action_pairs.png"),
    )
    env_plot.close()

    ############################################################################################################
    if args.source_mode == "safe":
        # Fine-tune the source policy to be safe in all critical states while preserving the learned trajectory
        print("  Finetuning source policy for safety …")
        finetuning_result_dct = finetune_policy(
            policy=source_actor,
            dataset=rashomon_dataset,
            env=make_frozenlake_env(env_map=env1_map, task_num=0, is_slippery=is_slippery),
            overlap_mode="policy",
            required_accuracy=1.0,
        )
        source_actor = finetuning_result_dct['policy']
        if not finetuning_result_dct['reached_target']:
            raise ValueError(
                f"Safety finetuning did not reach required accuracy. "
                f"Final accuracy: {finetuning_result_dct['final_accuracy']:.4f}, "
                f"required: {finetuning_result_dct['target_accuracy']:.4f}"
            )
    else:
        print("  Using original source policy (no safety finetuning)")

    ############################################################################################################

    n_safe_per_state = rashomon_dataset.tensors[1].sum(dim=1).tolist()
    max_safe_actions_per_state = max(n_safe_per_state) if n_safe_per_state else 0
    print(f"  Total safety-critical states: {len(n_safe_per_state)}")
    min_surrogate_threshold = max_safe_actions_per_state / (1 + max_safe_actions_per_state)

    safety_critical_states = [rashomon_dataset[i, :] for i in range(rashomon_dataset.tensors[0].shape[0])]

    # Find the smallest inverse_temp (>= 10) for which softmax mass on safe actions exceeds the surrogate threshold
    source_actor.eval()
    inverse_temp_start = 10
    with torch.no_grad():
        all_obs = rashomon_dataset.tensors[0]           # (N, obs_dim)
        safe_mask = rashomon_dataset.tensors[1]         # (N, N_ACTIONS) multi-hot
        logits = source_actor(all_obs)                  # (N, N_ACTIONS)
        safe_prob_mass = None
        for inverse_temp in range(inverse_temp_start, 1001):
            probs = torch.softmax(logits * inverse_temp, dim=1)
            safe_prob_mass = (probs * safe_mask).sum(dim=1)
            if safe_prob_mass.min().item() >= min_surrogate_threshold:
                break
        else:
            worst_idx = int(safe_prob_mass.argmin().item()) # type: ignore
            raise ValueError(
                f"Cannot find inverse_temp <= 1000 satisfying surrogate threshold. "
                f"Worst state {safety_critical_states[worst_idx]}: "
                f"safe-action softmax mass = {safe_prob_mass[worst_idx].item():.6f} < {min_surrogate_threshold:.6f}" # type: ignore
            )
    print(f"  Smallest inverse_temp satisfying surrogate threshold: {inverse_temp:.0f}")
    print(f"  Min safe-action softmax mass at T={inverse_temp:.0f}: {safe_prob_mass.min().item():.6f}")

    print(f"  Safety-critical states: {len(safety_critical_states)}")
    print(f"  Surrogate threshold: {min_surrogate_threshold:.6f}")

    min_hard_specification = 1.0
    _set_seeds(args.seed)
    interval_trainer = IntervalTrainer(
        model=source_actor,
        min_acc_limit=min_surrogate_threshold,
        seed=args.seed,
        n_iters=5_000, # args.rashomon_n_iters, # type: ignore
        min_acc_increment=0,
        T=inverse_temp,
        checkpoint=100, # TODO: save a checkpointed bound every checkpoint iterations # type: ignore
    )
    interval_trainer.compute_rashomon_set(
        dataset=rashomon_dataset, multi_label=True, aggregation='min' # type: ignore
    )
    final_certificate_idx = -1  # default: use last (largest) bounds
    if interval_trainer.certificates[-1] < min_hard_specification:
        # Last bounds don't satisfy spec — find the largest that does
        final_certificate_lst = [
            i for i, cert in enumerate(interval_trainer.certificates) if cert >= min_hard_specification
        ]
        if len(final_certificate_lst) == 0:
            raise ValueError(
                f"No Rashomon certificate satisfies the hard specification ({min_hard_specification}). "
                f"Best certificate: {max(interval_trainer.certificates):.4f}"
            )
        # Pick the last (largest bounds) among those that satisfy the spec
        final_certificate_idx = final_certificate_lst[-1]
    cert_hard = interval_trainer.certificates[final_certificate_idx]
    print(f"  Certified hard specification: {cert_hard:.4f}")

    bounded_model = interval_trainer.bounds[final_certificate_idx]
    param_bounds_l = [p.detach().cpu() for p in bounded_model.param_l]
    param_bounds_u = [p.detach().cpu() for p in bounded_model.param_u]

    # Save the bounded model for potential future analysis
    torch.save(bounded_model, downstream_dir / "bounded_model.pt") 

    # ── 3. SafeAdapt — PPO with Rashomon bounds ────────────────────────
    print("\n[3/6] SafeAdapt: PPO with Rashomon parameter bounds …")
    _set_seeds(args.seed)
    env2 = make_env(task=1)
    safe_cfg = PPOConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        ent_coef=args.ent_coef,
        early_stop=True,
        early_stop_reward_threshold=1.0,
        eval_episodes=args.eval_episodes,
    )
    safe_actor, _ = ppo_train( # type: ignore
        env=env2,
        cfg=safe_cfg,
        actor_warm_start=copy.deepcopy(source_actor),
        critic_warm_start=copy.deepcopy(source_critic),
        actor_param_bounds_l=param_bounds_l,
        actor_param_bounds_u=param_bounds_u,
    )
    env2.close()

    # ── 4. UnsafeAdapt — PPO without bounds ─────────────────────────────
    print("\n[4/6] UnsafeAdapt: PPO without any bounds …")
    _set_seeds(args.seed)
    env2 = make_env(task=1)
    unsafe_cfg = PPOConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        ent_coef=args.ent_coef,
        early_stop=True,
        early_stop_reward_threshold=1.0,
        eval_episodes=args.eval_episodes,
    )
    unsafe_actor, _ = ppo_train( # type: ignore
        env=env2,
        cfg=unsafe_cfg,
        actor_warm_start=copy.deepcopy(source_actor),
        critic_warm_start=copy.deepcopy(source_critic),
    )
    env2.close()

    # ── 5. EWC — PPO with EWC regularisation ───────────────────────────
    print("\n[5/6] EWC: Computing Fisher information and training …")
    _set_seeds(args.seed)
    ewc_state = compute_ewc_state(
        actor=copy.deepcopy(source_actor),
        observations=source_training_data["states"],
        compute_critic=False,
        device=args.device,
        fisher_sample_size=min(1000, len(source_training_data["states"])),
        seed=args.seed,
    )
    env2 = make_env(task=1)
    ewc_cfg = EWCPPOConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        ent_coef=args.ent_coef,
        early_stop=True,
        early_stop_reward_threshold=1.0,
        ewc_lambda=args.ewc_lambda,
        eval_episodes=args.eval_episodes,
    )
    ewc_actor, _ = ewc_ppo_train( # type: ignore
        env=env2,
        cfg=ewc_cfg,
        ewc_states=[ewc_state],
        actor_warm_start=copy.deepcopy(source_actor),
        critic_warm_start=copy.deepcopy(source_critic),
    )
    env2.close()

    # ── 6. Evaluate all policies ────────────────────────────────────────
    print("\n[6/6] Evaluating all policies …")

    # Ensure all actors are on CPU (ppo_train may leave them on CUDA)
    safe_actor.cpu()
    unsafe_actor.cpu()
    ewc_actor.cpu()

    # source_actor = _make_actor(obs_dim, n_actions, hidden=args.hidden)
    # source_actor.load_state_dict(
    #     torch.load(source_dir / "source_policy.pt", map_location="cpu")
    # )

    policies = {
        "Source":      source_actor,
        "SafeAdapt":   safe_actor,
        "UnsafeAdapt": unsafe_actor,
        "EWC":         ewc_actor,
    }

    rows: list[dict] = []
    for name, actor in policies.items():
        actor.eval()
        for task in (0, 1):
            env_eval = make_env(task)
            metrics = evaluate_policy(env_eval, actor, num_episodes=args.eval_episodes)
            env_eval.close()

            critical_state_safety_acc = None
            if task == 0:
                # Critical state safety accuracy on the Task-1 safety dataset
                critical_state_safety_acc, _ = verify_safety_posthoc(
                    actor=actor,
                    dataset=rashomon_dataset,
                    multi_label=True,
                    min_safety_limit=1.0,
                    env_map=env1_map,
                    verbose=False,
                    aggregation='mean', # mean -> avg safety flag per state -> safety "accuracy"
                )

            rows.append({
                "Policy": name,
                "Task": task + 1,
                "Trajectory Safety Rate": metrics["avg_safety_success"],
                "Critical State Safety Rate": critical_state_safety_acc,
                "Avg Total Reward": metrics["avg_total_reward"],
                "Success Rate": metrics["avg_success"],
                "Avg Steps": metrics["avg_steps"],
            })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    # Save table
    csv_path = out_dir / "results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTable saved to {csv_path}")

    # ── Save policy networks ───────────────────────────────────────────
    torch.save(safe_actor.state_dict(), downstream_dir / "safeadapt_actor.pt")
    torch.save(unsafe_actor.state_dict(), downstream_dir / "unsafeadapt_actor.pt")
    torch.save(ewc_actor.state_dict(), downstream_dir / "ewc_actor.pt")
    print("\nPolicy networks saved to downstream directory.")

    # ── Save trajectory plots ───────────────────────────────────────────
    print("\nSaving trajectory plots …")
    for name, actor in policies.items():
        actor.eval()
        for task in (0, 1):
            env_map = env1_map if task == 0 else env2_map
            render_env = make_frozenlake_env(
                env_map=env_map,
                task_num=task,
                is_slippery=is_slippery,
                render_mode="rgb_array",
            )
            save_path = plots_dir / f"{name}_Task{task + 1}_episode.png"
            plot_episode(
                actor=actor,
                env=render_env,
                deterministic=True,
                seed=args.seed,
                title=f"{name} — Task {task + 1}",
                save_path=str(save_path),
            )
            render_env.close()
            plt.close("all")

    # Save greedy-action overlay plots for Task 1
    print("Saving greedy-action overlay plots …")
    n_states = obs_dim - 1
    for name, actor in policies.items():
        actor.eval()
        sa_pairs = []
        for s in range(n_states):
            obs = np.zeros(obs_dim, dtype=np.float32)
            obs[s] = 1.0
            with torch.no_grad():
                action = actor(torch.tensor(obs).unsqueeze(0)).argmax(dim=1).item()
            sa_pairs.append((s, action))

        env_plot = make_frozenlake_env(
            env_map=env1_map, task_num=0,
            is_slippery=is_slippery, render_mode="rgb_array",
        )
        fig = plot_state_action_pairs(
            env=env_plot,
            state_action_pairs=sa_pairs,
            arrow_color="black",
            title=f"{name} — greedy actions (Task 1 map)",
            save_path=str(plots_dir / f"{name}_Task1_actions.png"),
        )
        env_plot.close()
        plt.close("all")

    print(f"\nAll plots saved to {plots_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
