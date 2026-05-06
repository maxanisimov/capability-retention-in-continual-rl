"""Static configuration for the FrozenLake safety pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.pipelines.frozenlake_safety.core.reference_settings import (
    LAYOUT,
    frozenlake_safety_diagonal_4x4_settings,
    settings_for_layout,
)


REFERENCE_SETTINGS = frozenlake_safety_diagonal_4x4_settings()
SOURCE_PPO_SETTINGS = REFERENCE_SETTINGS["source"]["ppo"]
ADAPTATION_PPO_SETTINGS = REFERENCE_SETTINGS["adaptation_ppo"]["ppo"]
ADAPTATION_EWC_SETTINGS = REFERENCE_SETTINGS["adaptation_ewc"]["ewc"]
ADAPTATION_RASHOMON_SETTINGS = REFERENCE_SETTINGS["adaptation_rashomon"]["rashomon"]
TASKS_SETTINGS = REFERENCE_SETTINGS["tasks"]
SETTINGS_SOURCE = (
    f"FrozenLake Safety {LAYOUT} "
    "(experiments/pipelines/frozenlake_safety/settings/source/train_source_policy_settings.yaml, "
    "experiments/pipelines/frozenlake_safety/settings/adaptation/{ppo,ewc,rashomon}.yaml, "
    "experiments/pipelines/frozenlake_safety/settings/tasks/envs.yaml)"
)
N_ACTIONS = 4
OBS_DIM = 3


@dataclass(frozen=True)
class PipelineConfig:
    layout: str = LAYOUT
    source_map: tuple[str, ...] = tuple(str(r) for r in TASKS_SETTINGS["source_map"])
    downstream_map: tuple[str, ...] = tuple(str(r) for r in TASKS_SETTINGS["downstream_map"])
    max_episode_steps: int = int(TASKS_SETTINGS["max_episode_steps"])
    source_task_num: float = float(REFERENCE_SETTINGS["adaptation_ppo"]["source_task_num"])
    downstream_task_num: float = float(REFERENCE_SETTINGS["adaptation_ppo"]["downstream_task_num"])
    reference_layout: str = LAYOUT
    reference_settings_source: str = SETTINGS_SOURCE
    reference_settings_files: dict[str, str] | None = None
    hidden: int = int(SOURCE_PPO_SETTINGS["hidden"])
    activation: str = str(REFERENCE_SETTINGS["source"].get("activation", "relu"))
    source_total_timesteps: int = int(SOURCE_PPO_SETTINGS["total_timesteps"])
    downstream_total_timesteps: int = int(ADAPTATION_PPO_SETTINGS["total_timesteps"])
    source_rollout_steps: int = int(SOURCE_PPO_SETTINGS["rollout_steps"])
    downstream_rollout_steps: int = int(ADAPTATION_PPO_SETTINGS["rollout_steps"])
    source_minibatch_size: int = int(SOURCE_PPO_SETTINGS["minibatch_size"])
    downstream_minibatch_size: int = int(ADAPTATION_PPO_SETTINGS["minibatch_size"])
    source_update_epochs: int = int(SOURCE_PPO_SETTINGS["update_epochs"])
    downstream_update_epochs: int = int(ADAPTATION_PPO_SETTINGS["update_epochs"])
    source_gamma: float = float(SOURCE_PPO_SETTINGS["gamma"])
    downstream_gamma: float = float(ADAPTATION_PPO_SETTINGS["gamma"])
    source_gae_lambda: float = float(SOURCE_PPO_SETTINGS["gae_lambda"])
    downstream_gae_lambda: float = float(ADAPTATION_PPO_SETTINGS["gae_lambda"])
    source_clip_coef: float = float(SOURCE_PPO_SETTINGS["clip_coef"])
    downstream_clip_coef: float = float(ADAPTATION_PPO_SETTINGS["clip_coef"])
    source_ent_coef: float = float(SOURCE_PPO_SETTINGS["ent_coef"])
    downstream_ent_coef: float = float(ADAPTATION_PPO_SETTINGS["ent_coef"])
    source_vf_coef: float = float(SOURCE_PPO_SETTINGS["vf_coef"])
    downstream_vf_coef: float = float(ADAPTATION_PPO_SETTINGS["vf_coef"])
    source_lr: float = float(SOURCE_PPO_SETTINGS["lr"])
    downstream_lr: float = float(ADAPTATION_PPO_SETTINGS["lr"])
    source_max_grad_norm: float = float(SOURCE_PPO_SETTINGS["max_grad_norm"])
    downstream_max_grad_norm: float = float(ADAPTATION_PPO_SETTINGS["max_grad_norm"])
    eval_episodes: int = 1
    source_early_stop_reward_threshold: float = float(SOURCE_PPO_SETTINGS["early_stop_reward_threshold"])
    source_early_stop_failure_rate_threshold: float = float(SOURCE_PPO_SETTINGS["early_stop_failure_rate_threshold"])
    downstream_early_stop_reward_threshold: float = float(
        ADAPTATION_PPO_SETTINGS["early_stop_deterministic_total_reward_threshold"],
    )
    early_stop_success_rate_threshold: float | None = None
    ewc_lambda: float = float(ADAPTATION_EWC_SETTINGS["ewc_lambda"])
    ewc_apply_to_critic: bool = bool(ADAPTATION_EWC_SETTINGS["ewc_apply_to_critic"])
    fisher_sample_size: int = int(ADAPTATION_EWC_SETTINGS["fisher_sample_size"])
    rashomon_settings_source: str = SETTINGS_SOURCE
    rashomon_total_timesteps: int = int(ADAPTATION_PPO_SETTINGS["total_timesteps"])
    rashomon_n_iters: int = int(ADAPTATION_RASHOMON_SETTINGS["rashomon_n_iters"])
    rashomon_checkpoint: int = int(ADAPTATION_RASHOMON_SETTINGS["rashomon_checkpoint"])
    inverse_temp_start: int = int(ADAPTATION_RASHOMON_SETTINGS["inverse_temp_start"])
    inverse_temp_max: int = int(ADAPTATION_RASHOMON_SETTINGS["inverse_temp_max"])
    rashomon_surrogate_aggregation: str = str(ADAPTATION_RASHOMON_SETTINGS["rashomon_surrogate_aggregation"])
    rashomon_min_hard_spec: float = float(ADAPTATION_RASHOMON_SETTINGS["rashomon_min_hard_spec"])
    safety_finetune_lr: float = float(ADAPTATION_RASHOMON_SETTINGS["safety_finetune_lr"])
    safety_finetune_max_epochs: int = int(ADAPTATION_RASHOMON_SETTINGS["safety_finetune_max_epochs"])


def get_pipeline_config(layout: str) -> PipelineConfig:
    s = settings_for_layout(layout)
    src_ppo = s["source"]["ppo"]
    adapt_ppo = s["adaptation_ppo"]["ppo"]
    adapt_ewc = s["adaptation_ewc"]["ewc"]
    adapt_rashomon = s["adaptation_rashomon"]["rashomon"]
    tasks = s["tasks"]
    settings_source = (
        f"FrozenLake Safety {layout} "
        "(experiments/pipelines/frozenlake_safety/settings/source/train_source_policy_settings.yaml, "
        "experiments/pipelines/frozenlake_safety/settings/adaptation/{ppo,ewc,rashomon}.yaml, "
        "experiments/pipelines/frozenlake_safety/settings/tasks/envs.yaml)"
    )
    return PipelineConfig(
        layout=layout,
        source_map=tuple(str(r) for r in tasks["source_map"]),
        downstream_map=tuple(str(r) for r in tasks["downstream_map"]),
        max_episode_steps=int(tasks["max_episode_steps"]),
        source_task_num=float(s["adaptation_ppo"]["source_task_num"]),
        downstream_task_num=float(s["adaptation_ppo"]["downstream_task_num"]),
        reference_layout=layout,
        reference_settings_source=settings_source,
        reference_settings_files=dict(s["settings_files"]),
        hidden=int(src_ppo["hidden"]),
        activation=str(s["source"].get("activation", "relu")),
        source_total_timesteps=int(src_ppo["total_timesteps"]),
        downstream_total_timesteps=int(adapt_ppo["total_timesteps"]),
        source_rollout_steps=int(src_ppo["rollout_steps"]),
        downstream_rollout_steps=int(adapt_ppo["rollout_steps"]),
        source_minibatch_size=int(src_ppo["minibatch_size"]),
        downstream_minibatch_size=int(adapt_ppo["minibatch_size"]),
        source_update_epochs=int(src_ppo["update_epochs"]),
        downstream_update_epochs=int(adapt_ppo["update_epochs"]),
        source_gamma=float(src_ppo["gamma"]),
        downstream_gamma=float(adapt_ppo["gamma"]),
        source_gae_lambda=float(src_ppo["gae_lambda"]),
        downstream_gae_lambda=float(adapt_ppo["gae_lambda"]),
        source_clip_coef=float(src_ppo["clip_coef"]),
        downstream_clip_coef=float(adapt_ppo["clip_coef"]),
        source_ent_coef=float(src_ppo["ent_coef"]),
        downstream_ent_coef=float(adapt_ppo["ent_coef"]),
        source_vf_coef=float(src_ppo["vf_coef"]),
        downstream_vf_coef=float(adapt_ppo["vf_coef"]),
        source_lr=float(src_ppo["lr"]),
        downstream_lr=float(adapt_ppo["lr"]),
        source_max_grad_norm=float(src_ppo["max_grad_norm"]),
        downstream_max_grad_norm=float(adapt_ppo["max_grad_norm"]),
        source_early_stop_reward_threshold=float(src_ppo["early_stop_reward_threshold"]),
        source_early_stop_failure_rate_threshold=float(src_ppo["early_stop_failure_rate_threshold"]),
        downstream_early_stop_reward_threshold=float(
            adapt_ppo["early_stop_deterministic_total_reward_threshold"],
        ),
        ewc_lambda=float(adapt_ewc["ewc_lambda"]),
        ewc_apply_to_critic=bool(adapt_ewc["ewc_apply_to_critic"]),
        fisher_sample_size=int(adapt_ewc["fisher_sample_size"]),
        rashomon_settings_source=settings_source,
        rashomon_total_timesteps=int(adapt_ppo["total_timesteps"]),
        rashomon_n_iters=int(adapt_rashomon["rashomon_n_iters"]),
        rashomon_checkpoint=int(adapt_rashomon["rashomon_checkpoint"]),
        inverse_temp_start=int(adapt_rashomon["inverse_temp_start"]),
        inverse_temp_max=int(adapt_rashomon["inverse_temp_max"]),
        rashomon_surrogate_aggregation=str(adapt_rashomon["rashomon_surrogate_aggregation"]),
        rashomon_min_hard_spec=float(adapt_rashomon["rashomon_min_hard_spec"]),
        safety_finetune_lr=float(adapt_rashomon["safety_finetune_lr"]),
        safety_finetune_max_epochs=int(adapt_rashomon["safety_finetune_max_epochs"]),
    )
