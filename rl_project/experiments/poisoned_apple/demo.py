"""
Poisoned Apple – Safe Continual Learning with SafeAdapt
=======================================================

Demonstrates safe continual learning in a gridworld where an agent must avoid
poisoned apples while collecting safe ones.  The challenge is adapting to a
distribution shift (Task 2) while **certifiably** maintaining safety on the
original task (Task 1).

Three strategies are compared:

1. **NoAdapt** – trained on Task 1 only, no adaptation.
2. **UnsafeAdapt** *(optional)* – adapted to Task 2 without constraints.
3. **SafeAdapt** – adapted to Task 2 with parameter-space bounds derived
   via interval-bound propagation on a safety dataset.

Configuration
-------------
All tuneable knobs are in the ``EXPERIMENT CONFIGURATION`` cell below.
Environment layouts and training hyper-parameters are loaded from
``demo_configs.yaml``.

Outputs
-------
* Console tables with safety / performance metrics.
* Trajectory visualisations (per-task grid plots).
* Parameter bound width plots.
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

# --- Which config from demo_configs.yaml ------------------------------------
cfg_name: str = "simple_8x8"

# --- Optional baselines ----------------------------------------------------
train_unsafe_adapt: bool = True  # train UnsafeAdapt (takes extra time)

# --- Compute device --------------------------------------------------------
device: str = "cpu"              # 'cpu' or 'cuda'

# --- Early stopping (applied to NoAdapt and SafeAdapt training) ------------
#   Checked at every periodic evaluation (~every 10×rollout_steps steps).
#   Triggers when ALL non-None thresholds are simultaneously satisfied.
#   Set early_stop=False to disable entirely.
early_stop: bool = True
early_stop_min_steps: int = 500
early_stop_reward_threshold: float | None = None
early_stop_failure_rate_threshold: float | None = 0.0  # stop if failure_rate <= this

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
import numpy as np
import pandas as pd
import torch
import yaml

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
_RL_DIR = os.path.join(_PROJECT_ROOT, "rl_project")
_FL_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "frozen_lake"))
for p in (_RL_DIR, _PROJECT_ROOT, _SCRIPT_DIR, _FL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from rl_project.utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer
from poisoned_apple_env import (
    PoisonedAppleEnv,
    evaluate_policy,
    get_all_unsafe_state_action_pairs,
    visualize_agent_trajectory,
)
from rl_project.experiments.frozen_lake.frozenlake_utils import (
    generate_sufficient_safe_state_action_dataset,
    set_all_seeds,
    train_safety_actor,
    verify_safety_accuracy,
)
from rl_project.utils.rashomon_utils import plot_parameter_bound_widths

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
grid_size: int = cfg["grid_size"]
max_steps: int = cfg["max_steps"]
observation_type: str = cfg["observation_type"]
agent_start_pos: tuple = tuple(cfg["agent_start_pos"])
seed: int = cfg["seed"]

env1_safe_apple_positions = [tuple(pos) for pos in cfg["env1_safe_apples"]]
env1_poisoned_apple_positions = [tuple(pos) for pos in cfg["env1_poisoned_apples"]]
env2_safe_apple_positions = [tuple(pos) for pos in cfg["env2_safe_apples"]]
env2_poisoned_apple_positions = [tuple(pos) for pos in cfg["env2_poisoned_apples"]]

no_adapt_timesteps: int = cfg["noadapt_max_train_timesteps"]
task2_adapt_timesteps: int = cfg.get("unsafeadapt_max_train_timesteps", 0)
safe_adapt_ppo_timesteps: int = cfg["safeadapt_max_train_timesteps"]

# ── Seed everything ────────────────────────────────────────────────────────
set_all_seeds(seed)

# ── Print summary ──────────────────────────────────────────────────────────
print("=" * 72)
print("POISONED APPLE – SAFE CONTINUAL LEARNING EXPERIMENT")
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

env1 = PoisonedAppleEnv(
    grid_size=grid_size,
    agent_start_pos=agent_start_pos,
    safe_apple_positions=env1_safe_apple_positions,
    poisoned_apple_positions=env1_poisoned_apple_positions,
    observation_type=observation_type,
    max_steps=max_steps,
    seed=seed,
)
env2 = PoisonedAppleEnv(
    grid_size=grid_size,
    agent_start_pos=agent_start_pos,
    safe_apple_positions=env2_safe_apple_positions,
    poisoned_apple_positions=env2_poisoned_apple_positions,
    observation_type=observation_type,
    max_steps=max_steps,
    seed=seed,
)

print(f"  Task 1:  safe apples={env1_safe_apple_positions}"
      f"  poisoned={env1_poisoned_apple_positions}")
print(f"  Task 2:  safe apples={env2_safe_apple_positions}"
      f"  poisoned={env2_poisoned_apple_positions}")


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

visualize_agent_trajectory(
    env1, standard_actor, num_episodes=1, max_steps=max_steps,
    env_name="Task 1", cfg_name=cfg_name, actor_name="NoAdapt",
    save_dir=_PLOTS_DIR if save_results else None,
)
visualize_agent_trajectory(
    env2, standard_actor, num_episodes=1, max_steps=max_steps,
    env_name="Task 2", cfg_name=cfg_name, actor_name="NoAdapt",
    save_dir=_PLOTS_DIR if save_results else None,
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

    visualize_agent_trajectory(
        env1, amnesic_actor, num_episodes=1, max_steps=max_steps,
        env_name="Task 1", cfg_name=cfg_name, actor_name="UnsafeAdapt",
        save_dir=_PLOTS_DIR if save_results else None,
    )
    visualize_agent_trajectory(
        env2, amnesic_actor, num_episodes=1, max_steps=max_steps,
        env_name="Task 2", cfg_name=cfg_name, actor_name="UnsafeAdapt",
        save_dir=_PLOTS_DIR if save_results else None,
    )
else:
    print("\n[3/9]  Skipping UnsafeAdapt  (train_unsafe_adapt=False)")


#%%
# =============================================================================
# 4. BUILD SAFETY DATASET
# =============================================================================
print("\n[4/9]  Building safety demonstration dataset …")

unsafe_pairs = get_all_unsafe_state_action_pairs(env=env1)
state_action_dataset = generate_sufficient_safe_state_action_dataset(unsafe_pairs, env1)
multi_label = True
print(f"\n  Safety demonstration dataset contains {len(state_action_dataset)} samples")


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

visualize_agent_trajectory(
    env1, safeadapt_actor, num_episodes=1, max_steps=max_steps,
    env_name="Task 1", cfg_name=cfg_name, actor_name="SafeAdapt",
    save_dir=_PLOTS_DIR if save_results else None,
)
visualize_agent_trajectory(
    env2, safeadapt_actor, num_episodes=1, max_steps=max_steps,
    env_name="Task 2", cfg_name=cfg_name, actor_name="SafeAdapt",
    save_dir=_PLOTS_DIR if save_results else None,
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
    env_map=None,
)
print(f"  Safety accuracy: {accuracy:.4f}  (meets requirement: {passed})")

#%%
# =============================================================================
# 9. FINAL EVALUATION & RESULTS TABLE
# =============================================================================
print("\n[9/9]  Evaluating all policies …")

num_eval_episodes = 1  # deterministic env + deterministic policy ⇒ 1 suffices

noadapt_t1 = evaluate_policy(env1, standard_actor, num_episodes=num_eval_episodes)
noadapt_t2 = evaluate_policy(env2, standard_actor, num_episodes=num_eval_episodes)
safeadapt_t1 = evaluate_policy(env1, safeadapt_actor, num_episodes=num_eval_episodes)
safeadapt_t2 = evaluate_policy(env2, safeadapt_actor, num_episodes=num_eval_episodes)

results = {
    "NoAdapt / Task 1": noadapt_t1,
    "NoAdapt / Task 2": noadapt_t2,
}
if train_unsafe_adapt and amnesic_actor is not None:
    results["UnsafeAdapt / Task 1"] = evaluate_policy(env1, amnesic_actor, num_episodes=num_eval_episodes)
    results["UnsafeAdapt / Task 2"] = evaluate_policy(env2, amnesic_actor, num_episodes=num_eval_episodes)
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
    for env, task_label in [(env1, "Task 1"), (env2, "Task 2")]:
        visualize_agent_trajectory(
            env, actor, num_episodes=1, max_steps=max_steps,
            env_name=task_label, cfg_name=cfg_name, actor_name=name,
            save_dir=_PLOTS_DIR if save_results else None,
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
