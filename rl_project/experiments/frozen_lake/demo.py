"""
FrozenLake – Safe Continual Learning with SafeAdapt
====================================================

Demonstrates safe continual learning in the Gymnasium FrozenLake environment.
An agent must navigate from start to goal while avoiding holes.  The challenge
is adapting to a distribution shift (Task 2) while **certifiably** maintaining
safety on the original task (Task 1).

Three strategies are compared:

1. **NoAdapt** – trained on Task 1 only, no adaptation.
2. **UnsafeAdapt** *(optional)* – adapted to Task 2 without constraints.
3. **SafeAdapt** – adapted to Task 2 with parameter-space bounds derived
   via interval-bound propagation on a safety dataset.

Configuration
-------------
All tuneable knobs are in the ``EXPERIMENT CONFIGURATION`` cell below.
Environment maps and training hyper-parameters are loaded from
``demo_configs.yaml``.

Outputs
-------
* Console tables with safety / performance metrics.
* Trajectory visualisations (single-task and multi-task grids).
* State-action pair plots (unsafe & sufficient-safe overlays).
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

# --- Which map / hyper-parameter set from demo_configs.yaml ----------------
cfg_name: str = "standard_4x4"

# --- Optional baselines ----------------------------------------------------
train_unsafe_adapt: bool = True  # train UnsafeAdapt (takes extra time)

# --- Compute device --------------------------------------------------------
device: str = "cpu"              # 'cpu' or 'cuda'

# --- Early stopping (applied to NoAdapt and SafeAdapt training) ------------
#   Checked at every periodic evaluation (~every 10×rollout_steps steps).
#   Triggers when ALL non-None thresholds are simultaneously satisfied.
#   Set early_stop=False to disable entirely.
early_stop: bool = True
early_stop_min_steps: int = 10_000      # do not check before this many steps
early_stop_reward_threshold: float | None = 1.0 # None  # stop if mean reward >= this
early_stop_failure_rate_threshold: float | None = 0.0 #  None  # stop if failure_rate <= this

# --- Output control ---------------------------------------------------------
save_results: bool = True        # write plots & tables to disk

# ─────────────────────────────────────────────────────────────────────────────
# End of user-configurable parameters.
# ─────────────────────────────────────────────────────────────────────────────

#%%
# =============================================================================
# IMPORTS
# =============================================================================
import os
import sys
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_RL_DIR = os.path.join(_PROJECT_ROOT, "rl_project")
for p in (_RL_DIR, _PROJECT_ROOT, _SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from rl_project.utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer
from utils.gymnasium_utils import (
    plot_gymnasium_episode,
    plot_gymnasium_episode_multitask,
    plot_state_action_pairs,
    plot_state_action_pairs_transition,
)
from frozenlake_utils import (
    build_all_safety_datasets,
    evaluate_policy,
    extract_position_action_pairs,
    generate_sufficient_safe_state_action_dataset,
    generate_sufficient_safe_state_action_dataset,
    get_all_unsafe_state_action_pairs,
    make_frozenlake_env,
    set_all_seeds,
    train_safety_actor,
    verify_safety_accuracy,
)
from utils.rashomon_utils import plot_parameter_bound_widths

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
# LOAD & VALIDATE CONFIGURATION
# =============================================================================
with open(os.path.join(_SCRIPT_DIR, "demo_configs.yaml")) as f:
    _ALL_CONFIGS = yaml.safe_load(f)

assert cfg_name in _ALL_CONFIGS, f"'{cfg_name}' not in demo_configs.yaml"
cfg = _ALL_CONFIGS[cfg_name]

# Unpack YAML fields
env1_map: list[str] = cfg["env1_map"]
env2_map: list[str] = cfg["env2_map"]
is_slippery: bool = cfg["is_slippery"]
max_steps: int = cfg["max_steps"]
seed: int = cfg["seed"]
no_adapt_timesteps: int = cfg["noadapt_max_train_timesteps"]
task2_adapt_timesteps: int = cfg.get("unsafeadapt_max_train_timesteps", 0)
safe_adapt_ppo_timesteps: int = cfg["safeadapt_max_train_timesteps"]

# ── Seed everything ────────────────────────────────────────────────────────
set_all_seeds(seed)

# ── Print summary ──────────────────────────────────────────────────────────
print("=" * 72)
print("FROZEN LAKE – SAFE CONTINUAL LEARNING EXPERIMENT")
print("=" * 72)
print(f"  Config           : {cfg_name}")
print(f"  Train UnsafeAdapt: {train_unsafe_adapt}")
print(f"  Save results     : {save_results}")
print(f"  Seed             : {seed}")
print(f"  Device           : {device}")
print(f"  Early stop       : {early_stop}"
      + (f"  (min_steps={early_stop_min_steps:,}, "
         f"reward>={early_stop_reward_threshold}, "
         f"failure_rate<={early_stop_failure_rate_threshold})" if early_stop else ""))
print("=" * 72)


#%%
# =============================================================================
# 1. CREATE ENVIRONMENTS
# =============================================================================
print("\n[1/9]  Creating environments …")

env1 = make_frozenlake_env(env1_map, task_num=0, is_slippery=is_slippery)
env2 = make_frozenlake_env(env2_map, task_num=1, is_slippery=is_slippery)

# rgb_array versions for plotting
env1_show = make_frozenlake_env(env1_map, task_num=0, is_slippery=is_slippery, render_mode="rgb_array")
env2_show = make_frozenlake_env(env2_map, task_num=1, is_slippery=is_slippery, render_mode="rgb_array")

print("  Task 1 map:")
for row in env1_map:
    print(f"    {row}")
print("  Task 2 map:")
for row in env2_map:
    print(f"    {row}")


#%%
# =============================================================================
# 2. TRAIN NoAdapt BASELINE (Task 1 only)
# =============================================================================
print(f"\n[2/9]  Training NoAdapt on Task 1  (max {no_adapt_timesteps:,} steps) …")

standard_actor, standard_critic, standard_training_data = ppo_train(
    env=env1,
    cfg=PPOConfig(
        total_timesteps=no_adapt_timesteps,
        device=device,
        early_stop=early_stop,
        early_stop_min_steps=early_stop_min_steps,
        early_stop_reward_threshold=early_stop_reward_threshold,
        early_stop_failure_rate_threshold=early_stop_failure_rate_threshold,
    ),
    return_training_data=True,
)
standard_actor.cpu()
standard_critic.cpu()
print("  NoAdapt training complete.")

# Visualise
_ = plot_gymnasium_episode(
    env=env1_show, actor=standard_actor,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} – Task 1 – NoAdapt",
    save_path=_save_path(f"{cfg_name}_Task1_NoAdapt.png"),
)
_ = plot_gymnasium_episode(
    env=env2_show, actor=standard_actor,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} – Task 2 – NoAdapt",
    save_path=_save_path(f"{cfg_name}_Task2_NoAdapt.png"),
)


#%%
# =============================================================================
# 3. (Optional) TRAIN UnsafeAdapt BASELINE
# =============================================================================
amnesic_actor = None
if train_unsafe_adapt:
    print(f"\n[3/9]  Training UnsafeAdapt on Task 2  (max {task2_adapt_timesteps:,} steps) …")
    amnesic_actor, _ = ppo_train(
        env=env2,
        cfg=PPOConfig(
            total_timesteps=task2_adapt_timesteps,
            device=device,
            early_stop=early_stop,
            early_stop_min_steps=early_stop_min_steps,
            early_stop_reward_threshold=early_stop_reward_threshold,
            early_stop_failure_rate_threshold=early_stop_failure_rate_threshold,
        ),
        actor_warm_start=standard_actor,
        critic_warm_start=standard_critic,
    )
    amnesic_actor.cpu()
    print("  UnsafeAdapt training complete.")

    _ = plot_gymnasium_episode(
        env=env1_show, actor=amnesic_actor,
        figsize_per_frame=(1.5, 1.5),
        title=f"{cfg_name} – Task 1 – UnsafeAdapt",
        save_path=_save_path(f"{cfg_name}_Task1_UnsafeAdapt.png"),
    )
    _ = plot_gymnasium_episode(
        env=env2_show, actor=amnesic_actor,
        figsize_per_frame=(1.5, 1.5),
        title=f"{cfg_name} – Task 2 – UnsafeAdapt",
        save_path=_save_path(f"{cfg_name}_Task2_UnsafeAdapt.png"),
    )
else:
    print("\n[3/9]  Skipping UnsafeAdapt  (train_unsafe_adapt=False)")


#%%
# =============================================================================
# 4. BUILD SAFETY DATASET
# =============================================================================
print("\n[4/9]  Building safety demonstration dataset …")

unsafe_pairs = get_all_unsafe_state_action_pairs(env_map=env1_map, task_num=0)
state_action_dataset = generate_sufficient_safe_state_action_dataset(unsafe_pairs, env1)
multi_label = True
print(f"\n Safety demonstration dataset contains {len(state_action_dataset)} samples")

#%%
# ── Visualise unsafe & sufficient-safe pairs on the grid ────────────────────
unsafe_pos_pairs = get_all_unsafe_state_action_pairs(env1_map, task_num=0, state_repr="position")
safe_pos_pairs = extract_position_action_pairs(state_action_dataset)

env1_vis = gym.make("FrozenLake-v1", desc=env1_map, is_slippery=is_slippery, render_mode="rgb_array")

# _ = plot_state_action_pairs(
#     env=env1_vis, state_action_pairs=unsafe_pos_pairs,
#     title="Unsafe State-Action Pairs", arrow_color="red",
#     save_path=_save_path(f"{cfg_name}_unsafe_pairs.png"),
# )
# _ = plot_state_action_pairs(
#     env=env1_vis, state_action_pairs=safe_pos_pairs,
#     title="Sufficient Safe State-Action Pairs", arrow_color="green",
#     save_path=_save_path(f"{cfg_name}_sufficient_safe_pairs.png"),
# )

_ = plot_state_action_pairs_transition(
    env_left=env1_vis, 
    env_right=env1_vis,
    pairs_left=unsafe_pos_pairs,
    pairs_right=safe_pos_pairs,
    title_left="Unsafe Pairs", 
    title_right="Sufficient Safe Pairs",
    # save_path=_save_path(f"{cfg_name}_transition.png"),
)


#%%
# =============================================================================
# 5. TRAIN SAFETY ACTOR (reference model)
# =============================================================================
print("\n[5/9]  Training base safety actor …")

safety_actor, safety_acc = train_safety_actor(
    base_actor=standard_actor,
    dataset=state_action_dataset,
    multi_label=multi_label,
    device=device,
    use_margin_loss=True,
    margin=14.0,
)

assert safety_acc == 1.0, (
    f"Base safety actor must achieve perfect accuracy on the safety dataset "
    f"(got {safety_acc:.4f})"
)
min_acc_limit = safety_acc


#%%
# =============================================================================
# 6. COMPUTE Rashomon set (parameter-space bounds via IBP)
# =============================================================================
print(f"\n[6/9]  Computing Rashomon bounds  (min_acc={min_acc_limit:.2f}) …")

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
if not certificate >= min_acc_limit:
    print(f"  WARNING: Certificate does not meet minimum accuracy requirement "
          f"(got {certificate:.4f}, required {min_acc_limit:.4f})")
    # raise AssertionError("Certificate does not meet minimum accuracy requirement")
print(f"  Parameter layers constrained: {len(param_bounds_l)}")

plot_parameter_bound_widths(
    param_bounds_l=param_bounds_l,
    param_bounds_u=param_bounds_u,
    layer_names=None,  # auto-generate names like W0, b0, W1, b1, ...
    title=f"{cfg_name} – Rashomon Parameter Bounds",
    figsize=(10, 6),
    save_path=_save_path(f"{cfg_name}_rashomon_bounds.png"),
    log_scale=False,
)

#%%
# =============================================================================
# 7. TRAIN SafeAdapt (Task 2 with parameter constraints)
# =============================================================================
print(f"\n[7/9]  Training SafeAdapt on Task 2  (max {safe_adapt_ppo_timesteps:,} steps) …")

safeadapt_actor, _ = ppo_train(
    env=env2,
    cfg=PPOConfig(
        total_timesteps=safe_adapt_ppo_timesteps,
        device=device,
        early_stop=early_stop,
        early_stop_min_steps=early_stop_min_steps,
        early_stop_reward_threshold=early_stop_reward_threshold,
        early_stop_failure_rate_threshold=early_stop_failure_rate_threshold,
    ),
    actor_warm_start=standard_actor,
    critic_warm_start=standard_critic,
    actor_param_bounds_l=param_bounds_l,
    actor_param_bounds_u=param_bounds_u,
)
safeadapt_actor.cpu()
print("  SafeAdapt training complete.")

_ = plot_gymnasium_episode(
    env=env1_show, actor=safeadapt_actor,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} – Task 1 – SafeAdapt",
    save_path=_save_path(f"{cfg_name}_Task1_SafeAdapt.png"),
)
_ = plot_gymnasium_episode(
    env=env2_show, actor=safeadapt_actor,
    figsize_per_frame=(1.5, 1.5),
    title=f"{cfg_name} – Task 2 – SafeAdapt",
    save_path=_save_path(f"{cfg_name}_Task2_SafeAdapt.png"),
)


#%%
# =============================================================================
# 8. VERIFY SAFETY CERTIFICATE
# =============================================================================
print("\n[8/9]  Verifying safety certificate on SafeAdapt actor …")

accuracy, passed = verify_safety_accuracy(
    actor=safeadapt_actor,
    dataset=state_action_dataset,
    multi_label=multi_label,
    min_acc_limit=min_acc_limit,
    env_map=env1_map,
)


#%%
# =============================================================================
# 9. FINAL EVALUATION & RESULTS TABLE
# =============================================================================
print("\n[9/9]  Evaluating all policies …")

num_eval_episodes = 1  # deterministic env + deterministic policy ⇒ 1 suffices

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
    csv_name = f"results_{cfg_name}.csv"
    csv_path = os.path.join(_TABLES_DIR, csv_name)
    results_df.to_csv(csv_path)
    print(f"\nResults table saved to: {csv_path}")


#%%
# =============================================================================
# MULTI-TASK TRAJECTORY VISUALISATIONS
# =============================================================================
print("\nGenerating multi-task trajectory plots …")

actors_to_plot = {"NoAdapt": standard_actor, "SafeAdapt": safeadapt_actor}
if train_unsafe_adapt and amnesic_actor is not None:
    actors_to_plot["UnsafeAdapt"] = amnesic_actor

for name, actor in actors_to_plot.items():
    _ = plot_gymnasium_episode_multitask(
        env_task1=env1_show,
        env_task2=env2_show,
        actor=actor,
        n_cols=7,
        figsize_per_frame=(1.5, 1.5),
        title=f"{cfg_name} – {name} on Task 1 & Task 2",
        save_path=_save_path(f"{cfg_name}_{name}_multitask.png"),
        one_row_per_task=True,
    )


#%%
# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 72)
print("EXPERIMENT COMPLETE")
print("=" * 72)
print(f"  Config         : {cfg_name}")
print(f"  Seed           : {seed}")
if save_results:
    print(f"  Plots saved to : {_PLOTS_DIR}")
    print(f"  Tables saved to: {_TABLES_DIR}")
print("=" * 72)

#%%