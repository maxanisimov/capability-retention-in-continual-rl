"""
Highway Parking (Discrete SAC) – Safe Continual Learning with SafeAdapt
========================================================================

Demonstrates safe continual learning in the highway-env Parking environment
using a **discrete Soft Actor-Critic** (SAC) agent.  The continuous 2-D
action space (acceleration × steering) is discretised into a 5×5 grid of
25 actions.

An ego vehicle must park in a designated spot while avoiding collisions with
parked vehicles.  The challenge is adapting to a distribution shift (Task 2 —
different goal spot and obstacle layout) while **certifiably** maintaining
safety on the original task (Task 1).

Three strategies are compared:

1. **NoAdapt** – trained on Task 1 only (discrete SAC), no adaptation.
2. **UnsafeAdapt** *(optional)* – adapted to Task 2 without constraints
   (discrete SAC warm-started from NoAdapt).
3. **SafeAdapt** – adapted to Task 2 with parameter-space bounds derived
   via interval-bound propagation on a safety dataset (discrete SAC + PGD).

Configuration
-------------
All tuneable knobs are in the ``EXPERIMENT CONFIGURATION`` cell below.

Outputs
-------
* Console tables with reward / success / safety metrics.
* Trajectory visualisations (single-task and multi-task grids).
* When ``save_results = True`` all plots are written to ``plots/`` and
  metrics tables to ``tables/``.
"""

#%%
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      EXPERIMENT CONFIGURATION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Edit only this cell to change the experiment.
# Everything below should run without modification.
# ─────────────────────────────────────────────────────────────────────────────

# --- Discretisation grid ---------------------------------------------------
n_bins_accel: int = 5
n_bins_steer: int = 5

# --- Which safety dataset to use ------------------------------------------
#   'Safe Optimal Policy Data'  – rollouts of the trained NoAdapt actor
#   'Safe Training Data'        – filtered transitions from SAC training
safe_state_action_data_name: str = "Safe Training Data"

# --- Training hyper-parameters ---------------------------------------------
noadapt_timesteps: int = 130_000       # NoAdapt (Task 1) training steps
unsafeadapt_timesteps: int = 130_000   # UnsafeAdapt (Task 2) training steps
safeadapt_timesteps: int = 130_000     # SafeAdapt (Task 2 + PGD) training steps
safe_env1_rollouts: int = 100          # rollouts for Safe Optimal Policy Data

# --- SafeAdapt bounds ------------------------------------------------------
safeadapt_min_acc: float = 0.99        # Rashomon set certified-accuracy target

# --- Optional baselines ----------------------------------------------------
train_unsafe_adapt: bool = True        # train UnsafeAdapt (takes extra time)

# --- Compute device --------------------------------------------------------
device: str = "cuda"                    # 'cpu' or 'cuda'

# --- Output control ---------------------------------------------------------
save_results: bool = False             # write plots & tables to disk
num_eval_episodes: int = 100           # episodes for policy evaluation
seed: int = 42

# ─────────────────────────────────────────────────────────────────────────────
# End of user-configurable parameters.
# ─────────────────────────────────────────────────────────────────────────────


#%%
# =============================================================================
# IMPORTS & PATH SETUP
# =============================================================================
import os
import sys

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_RL_DIR = os.path.join(_PROJECT_ROOT, "rl_project")
for p in (_RL_DIR, _PROJECT_ROOT, _SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import highway_env  # noqa: F401 — registers highway-env envs
from utils.custom_envs import CustomParkingEnv  # noqa: F401 — registers custom-parking-v0

from utils.discrete_sac_utils import (
    DiscreteSACConfig,
    discrete_sac_train,
)
from src.trainer import IntervalTrainer
from utils.gymnasium_utils import (
    plot_gymnasium_episode,
    plot_gymnasium_episode_multitask,
)
from parking_utils import (
    build_safety_datasets,
    evaluate_policy,
    make_discrete_parking_env,
    set_all_seeds,
    train_safety_actor,
    verify_safety_accuracy,
)

# ── Output directories ─────────────────────────────────────────────────────
_PLOTS_DIR = os.path.join(_SCRIPT_DIR, "plots")
_TABLES_DIR = os.path.join(_SCRIPT_DIR, "tables")
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_TABLES_DIR, exist_ok=True)


def _save_path(filename: str) -> str | None:
    """Return full path inside plots/ when saving is enabled, else None."""
    if not save_results:
        return None
    return os.path.join(_PLOTS_DIR, filename)


#%%
# =============================================================================
# ENVIRONMENT CONFIGURATIONS
# =============================================================================
# Both tasks share the same base settings; only goal / obstacle placement
# differs.  The action type is ContinuousAction — the DiscretizeActionWrapper
# added by make_discrete_parking_env converts it to a 5×5 discrete grid.

_ENV_BASE = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
    "success_goal_reward": 0.12,
    "collision_reward": -5,
    "steering_range": np.deg2rad(45),
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 100,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "controlled_vehicles": 1,
    "add_walls": True,
}

env_config_task1 = {
    **_ENV_BASE,
    "goal_spots": [("b", "c", 1)],
    "vehicles_count": 4,
    "parked_vehicles_spots": [
        ("a", "b", 0), ("a", "b", 1),
        ("b", "c", 0), ("b", "c", 2),
    ],
}

env_config_task2 = {
    **_ENV_BASE,
    "goal_spots": [("a", "b", 10)],
    "vehicles_count": 5,
    "parked_vehicles_spots": [
        ("a", "b", 9), ("a", "b", 11),
        ("b", "c", 9), ("b", "c", 10), ("b", "c", 11),
    ],
}


#%%
# =============================================================================
# DERIVED CONSTANTS & SEED
# =============================================================================
n_actions = n_bins_accel * n_bins_steer

assert safe_state_action_data_name in (
    "Safe Optimal Policy Data",
    "Safe Training Data",
), f"Invalid safety dataset: {safe_state_action_data_name}"

set_all_seeds(seed)

# ── GPU acceleration (auto-configured based on device) ──────────────────────
# When device == "cuda": 4 parallel envs, larger minibatch, torch.compile.
# When device == "cpu":  single env, default minibatch, no compilation.
_num_envs      = 4     if device == "cuda" else 1
_batch_size    = 1024  if device == "cuda" else 256
_compile_model = device == "cuda"

# Common SAC config shared across all training runs
_sac_cfg_base = dict(
    seed=seed,
    eval_episodes=10,
    buffer_size=500_000,
    batch_size=_batch_size,
    num_envs=_num_envs,
    compile_model=_compile_model,
    device=device,
    gamma=0.99,
    tau=0.005,
    lr=3e-4,
    alpha=0.05,
    autotune_alpha=False,
    policy_frequency=2,
    target_network_frequency=1,
    learning_starts=10_000,
)


def _make_env(config, render_mode=None):
    """Shorthand: create a discrete-action parking env (no task indicator)."""
    return make_discrete_parking_env(
        env_config=config,
        task_num=None,  # 12-dim obs (no task indicator)
        n_bins_accel=n_bins_accel,
        n_bins_steer=n_bins_steer,
        render_mode=render_mode,
    )


# ── Print summary ──────────────────────────────────────────────────────────
print("=" * 72)
print("HIGHWAY PARKING (DISCRETE SAC) – SAFE CONTINUAL LEARNING")
print("=" * 72)
print(f"  Grid             : {n_bins_accel}x{n_bins_steer}  ({n_actions} actions)")
print(f"  Safety dataset   : {safe_state_action_data_name}")
print(f"  NoAdapt steps    : {noadapt_timesteps:,}")
print(f"  SafeAdapt steps  : {safeadapt_timesteps:,}")
print(f"  Min acc target   : {safeadapt_min_acc}")
print(f"  Train UnsafeAdapt: {train_unsafe_adapt}")
print(f"  Eval episodes    : {num_eval_episodes}")
print(f"  Save results     : {save_results}")
print(f"  Seed             : {seed}")
print(f"  Device           : {device}")
print(f"  Num envs         : {_num_envs}  (vectorised: {_num_envs > 1})")
print(f"  Batch size       : {_batch_size}")
print(f"  torch.compile    : {_compile_model}")
print("=" * 72)


#%%
# =============================================================================
# 1. CREATE ENVIRONMENTS
# =============================================================================
print("\n[1/9]  Creating environments ...")

env1 = _make_env(env_config_task1)
env2 = _make_env(env_config_task2)

# rgb_array versions for plotting
env1_show = _make_env(env_config_task1, render_mode="rgb_array")
env2_show = _make_env(env_config_task2, render_mode="rgb_array")

print(f"  Task 1 goal spots : {env_config_task1['goal_spots']}")
print(f"  Task 1 obstacles  : {env_config_task1['parked_vehicles_spots']}")
print(f"  Task 2 goal spots : {env_config_task2['goal_spots']}")
print(f"  Task 2 obstacles  : {env_config_task2['parked_vehicles_spots']}")
print(f"  Obs dim           : {env1.observation_space.shape[0]}")
print(f"  Action space      : Discrete({n_actions})")


#%%
# =============================================================================
# 2. TRAIN NoAdapt BASELINE (Task 1 only, Discrete SAC)
# =============================================================================
print(f"\n[2/9]  Training NoAdapt on Task 1  ({noadapt_timesteps:,} steps, Discrete SAC) ...")

start_time = torch.cuda.Event(enable_timing=True)
standard_actor, standard_qf1, standard_qf2, standard_training_data = discrete_sac_train(
    env=lambda: _make_env(env_config_task1),
    cfg=DiscreteSACConfig(**_sac_cfg_base, total_timesteps=noadapt_timesteps),
    return_training_data=True,
)
end_time = torch.cuda.Event(enable_timing=True)
print(f"  NoAdapt training complete.  Time taken: {start_time.elapsed_time(end_time)/1000:.2f} seconds")
standard_actor.cpu()
standard_qf1.cpu()
standard_qf2.cpu()
print("  NoAdapt training complete.")

# Visualise
_ = plot_gymnasium_episode(
    env=env1_show, actor=standard_actor.cpu(),
    figsize_per_frame=(3.0, 3.0),
    title="Discrete SAC - Task 1 - NoAdapt",
    save_path=_save_path("disc_sac_Task1_NoAdapt.png"),
)
_ = plot_gymnasium_episode(
    env=env2_show, actor=standard_actor.cpu(),
    figsize_per_frame=(3.0, 3.0),
    title="Discrete SAC - Task 2 - NoAdapt",
    save_path=_save_path("disc_sac_Task2_NoAdapt.png"),
)

#%%
# =============================================================================
# 3. (Optional) TRAIN UnsafeAdapt BASELINE (Task 2, Discrete SAC, warm-start)
# =============================================================================
amnesic_actor = None
if train_unsafe_adapt:
    print(f"\n[3/9]  Training UnsafeAdapt on Task 2  ({unsafeadapt_timesteps:,} steps, Discrete SAC) ...")
    amnesic_actor, _, _ = discrete_sac_train(
        env=lambda: _make_env(env_config_task2),
        cfg=DiscreteSACConfig(**_sac_cfg_base, total_timesteps=unsafeadapt_timesteps),
        actor_warm_start=standard_actor,
        critic_warm_start=standard_qf1,
    )
    amnesic_actor.cpu()
    print("  UnsafeAdapt training complete.")

    _ = plot_gymnasium_episode(
        env=env1_show, actor=amnesic_actor.cpu(),
        figsize_per_frame=(3.0, 3.0),
        title="Discrete SAC - Task 1 - UnsafeAdapt",
        save_path=_save_path("disc_sac_Task1_UnsafeAdapt.png"),
    )
    _ = plot_gymnasium_episode(
        env=env2_show, actor=amnesic_actor.cpu(),
        figsize_per_frame=(3.0, 3.0),
        title="Discrete SAC - Task 2 - UnsafeAdapt",
        save_path=_save_path("disc_sac_Task2_UnsafeAdapt.png"),
    )
else:
    print("\n[3/9]  Skipping UnsafeAdapt  (train_unsafe_adapt=False)")


#%%
# =============================================================================
# 4. BUILD SAFETY DATASETS
# =============================================================================
print("\n[4/9]  Building safety datasets ...")

safe_datasets = build_safety_datasets(
    env=_make_env(env_config_task1),
    actor=standard_actor,
    training_data=standard_training_data,
    num_rollouts=safe_env1_rollouts,
    seed=seed,
)

state_action_dataset = safe_datasets[safe_state_action_data_name]
# Safe Training Data may have multiple valid actions per state → multi-label
# Safe Optimal Policy Data uses deterministic rollouts → single-label
multi_label = safe_state_action_data_name != "Safe Optimal Policy Data"
print(f"\n  Selected: {safe_state_action_data_name}  ({len(state_action_dataset)} samples)")
print(f"  Multi-label: {multi_label}")


#%%
# =============================================================================
# 5. TRAIN SAFETY ACTOR (reference model)
# =============================================================================
print("\n[5/9]  Training safety actor ...")

safety_actor, safety_acc = train_safety_actor(
    base_actor=standard_actor,
    dataset=state_action_dataset,
)
print(f"  Safety actor accuracy: {safety_acc:.4f}")

min_acc_limit = safeadapt_min_acc


#%%
# =============================================================================
# 6. COMPUTE SafeAdapt SET (parameter-space bounds via IBP)
# =============================================================================
print(f"\n[6/9]  Computing SafeAdapt bounds  (min_acc={min_acc_limit:.2f}) ...")

interval_trainer = IntervalTrainer(
    model=safety_actor.cpu(),
    min_acc_limit=min_acc_limit,
    seed=seed,
)
interval_trainer.compute_rashomon_set(
    dataset=state_action_dataset,
    multi_label=multi_label,
)

assert len(interval_trainer.bounds) == 1, "Expected exactly one bounded model"
certificate = interval_trainer.certificates[0]
bounded_model = interval_trainer.bounds[0]
param_bounds_l = [b.detach().cpu() for b in bounded_model.param_l]
param_bounds_u = [b.detach().cpu() for b in bounded_model.param_u]

print(f"  Certified accuracy: {certificate:.4f}")
print(f"  Parameter layers constrained: {len(param_bounds_l)}")


#%%
# =============================================================================
# 7. TRAIN SafeAdapt (Task 2 with parameter constraints, Discrete SAC + PGD)
# =============================================================================
print(f"\n[7/9]  Training SafeAdapt on Task 2  ({safeadapt_timesteps:,} steps, Discrete SAC + PGD) ...")

safeadapt_actor, _, _ = discrete_sac_train(
    env=lambda: _make_env(env_config_task2),
    cfg=DiscreteSACConfig(**_sac_cfg_base, total_timesteps=safeadapt_timesteps),
    actor_warm_start=standard_actor,
    critic_warm_start=standard_qf1,
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u,
)
safeadapt_actor.cpu()
print("  SafeAdapt training complete.")

_ = plot_gymnasium_episode(
    env=env1_show, actor=safeadapt_actor,
    figsize_per_frame=(3.0, 3.0),
    title="Discrete SAC - Task 1 - SafeAdapt",
    save_path=_save_path("disc_sac_Task1_SafeAdapt.png"),
)
_ = plot_gymnasium_episode(
    env=env2_show, actor=safeadapt_actor,
    figsize_per_frame=(3.0, 3.0),
    title="Discrete SAC - Task 2 - SafeAdapt",
    save_path=_save_path("disc_sac_Task2_SafeAdapt.png"),
)


#%%
# =============================================================================
# 8. VERIFY SAFETY CERTIFICATE
# =============================================================================
print("\n[8/9]  Verifying safety certificate on SafeAdapt actor ...")

accuracy, passed = verify_safety_accuracy(
    actor=safeadapt_actor,
    dataset=state_action_dataset,
    min_acc_limit=min_acc_limit,
)


#%%
# =============================================================================
# 9. FINAL EVALUATION & RESULTS TABLE
# =============================================================================
print("\n[9/9]  Evaluating all policies ...")

noadapt_t1 = evaluate_policy(env1, standard_actor, num_eval_episodes)
noadapt_t2 = evaluate_policy(env2, standard_actor, num_eval_episodes)
safeadapt_t1 = evaluate_policy(env1, safeadapt_actor, num_eval_episodes)
safeadapt_t2 = evaluate_policy(env2, safeadapt_actor, num_eval_episodes)

results = {
    "NoAdapt / Task 1": noadapt_t1,
    "NoAdapt / Task 2": noadapt_t2,
}
if train_unsafe_adapt and amnesic_actor is not None:
    results["UnsafeAdapt / Task 1"] = evaluate_policy(env1, amnesic_actor, num_eval_episodes)
    results["UnsafeAdapt / Task 2"] = evaluate_policy(env2, amnesic_actor, num_eval_episodes)
results["SafeAdapt / Task 1"] = safeadapt_t1
results["SafeAdapt / Task 2"] = safeadapt_t2

results_df = pd.DataFrame(results)

# ── Print ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("RESULTS SUMMARY")
print("=" * 72)
print(results_df.round(4).to_string())

# ── Verify key property ───────────────────────────────────────────────────
safe_t1 = results_df.loc["avg_safety_success", "SafeAdapt / Task 1"]
print(f"\nSafeAdapt perfect safety on Task 1: {safe_t1 >= min_acc_limit}  (safety={safe_t1:.3f})")

# ── Save ───────────────────────────────────────────────────────────────────
if save_results:
    csv_name = f"results_disc_sac_{safe_state_action_data_name.replace(' ', '_')}.csv"
    csv_path = os.path.join(_TABLES_DIR, csv_name)
    results_df.to_csv(csv_path)
    print(f"\nResults table saved to: {csv_path}")


#%%
# =============================================================================
# MULTI-TASK TRAJECTORY VISUALISATIONS
# =============================================================================
print("\nGenerating multi-task trajectory plots ...")

actors_to_plot = {"NoAdapt": standard_actor, "SafeAdapt": safeadapt_actor}
if train_unsafe_adapt and amnesic_actor is not None:
    actors_to_plot["UnsafeAdapt"] = amnesic_actor

for name, actor in actors_to_plot.items():
    _ = plot_gymnasium_episode_multitask(
        env_task1=env1_show,
        env_task2=env2_show,
        actor=actor,
        n_cols=7,
        figsize_per_frame=(3.0, 3.0),
        title=f"Discrete SAC - {name} on Task 1 & Task 2",
        save_path=_save_path(f"disc_sac_{name}_multitask.png"),
    )


#%%
# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 72)
print("EXPERIMENT COMPLETE")
print("=" * 72)
print(f"  Grid             : {n_bins_accel}x{n_bins_steer}")
print(f"  Safety dataset   : {safe_state_action_data_name}")
print(f"  NoAdapt steps    : {noadapt_timesteps:,}")
print(f"  SafeAdapt steps  : {safeadapt_timesteps:,}")
print(f"  Certified acc.   : {certificate:.4f}")
print(f"  Safety verified  : {passed}")
print(f"  Seed             : {seed}")
if save_results:
    print(f"  Plots saved to   : {_PLOTS_DIR}")
    print(f"  Tables saved to  : {_TABLES_DIR}")
print("=" * 72)

#%%
