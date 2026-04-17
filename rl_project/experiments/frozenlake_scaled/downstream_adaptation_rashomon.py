"""Adapt source policy to downstream scaled FrozenLake via Rashomon-constrained PPO."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import os
os.environ["SDL_AUDIODRIVER"] = "dummy" # to disable ALSA warnings when running on headless servers without audio devices
import numpy as np
import torch
from torch.utils.data import TensorDataset
import yaml

from rl_project.experiments.frozenlake_scaled.train_source_policy import build_actor_critic, make_env_from_layout
from rl_project.utils.gymnasium_utils import plot_episode
from rl_project.utils.ppo_utils import PPOConfig, evaluate, ppo_train
from src.trainer.IntervalTrainer import IntervalTrainer


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _find_layout_with_ppo(layout: str, current_file: Path) -> Path | None:
    """Return an alternative settings file that contains a PPO block for this layout."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "settings" / "downstream_adaptation_settings_ppo.yaml",
    ]
    current_resolved = current_file.resolve()
    for candidate in candidates:
        if not candidate.exists() or candidate.resolve() == current_resolved:
            continue
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        layout_cfg = data.get(layout, None)
        if isinstance(layout_cfg, dict) and isinstance(layout_cfg.get("ppo", None), dict):
            return candidate
    return None


def neutralize_task_feature(
    model: torch.nn.Sequential,
    task_feature_index: int,
    target_task_value: float,
) -> None:
    """Neutralize first-layer task feature contribution for target task value."""
    first = model[0]
    if not isinstance(first, torch.nn.Linear):
        raise ValueError("Expected first layer to be torch.nn.Linear for task-feature neutralization.")

    with torch.no_grad():
        w_task = first.weight[:, task_feature_index].clone()
        first.bias[:] = first.bias - w_task * target_task_value
        first.weight[:, task_feature_index] = 0.0


def create_source_trajectory_rashomon_dataset(
    actor: torch.nn.Module,
    env,
    *,
    seed: int,
    n_actions: int,
) -> tuple[TensorDataset, list[dict[str, int]]]:
    """Roll out source policy and create one-action-per-sample Rashomon dataset."""
    actor.eval()
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    nrow, ncol = env.unwrapped.desc.shape

    obs_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    state_action_pairs: list[dict[str, int]] = []

    while not done:
        obs_np = np.asarray(obs, dtype=np.float32).copy()
        obs_list.append(obs_np)

        with torch.no_grad():
            logits = actor(torch.from_numpy(obs_np).unsqueeze(0))
            action = int(torch.argmax(logits, dim=1).item())

        action_mask = np.zeros(n_actions, dtype=np.float32)
        action_mask[action] = 1.0
        label_list.append(action_mask)

        state_idx = int(env.unwrapped.s)
        row = state_idx // ncol
        col = state_idx % ncol
        state_action_pairs.append(
            {
                "step": int(step),
                "state_index": int(state_idx),
                "row": int(row),
                "col": int(col),
                "action": int(action),
            },
        )

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    if not obs_list:
        raise RuntimeError("Source trajectory is empty; cannot build Rashomon dataset.")

    obs_tensor = torch.tensor(np.asarray(obs_list), dtype=torch.float32)
    label_tensor = torch.tensor(np.asarray(label_list), dtype=torch.float32)
    return TensorDataset(obs_tensor, label_tensor), state_action_pairs


def _certificate_to_float(certificate: object) -> float:
    if certificate is None:
        return float("-inf")
    if isinstance(certificate, list):
        vals = [float(v) for v in certificate if v is not None]
        return min(vals) if vals else float("-inf")
    return float(certificate) # type: ignore


def compute_rashomon_bounds(
    *,
    actor: torch.nn.Module,
    rashomon_dataset: TensorDataset,
    seed: int,
    rashomon_n_iters: int,
    # min_acc_limit: float,
    aggregation: str,
    inverse_temp_start: int,
    inverse_temp_max: int,
    checkpoint: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, int, float, list[float], int]:
    """Compute Rashomon parameter bounds for a trajectory dataset with 100% accuracy on it."""
    if len(rashomon_dataset) == 0:
        raise RuntimeError("Rashomon dataset is empty.")

    n_valid_actions = rashomon_dataset.tensors[1].sum(dim=1).tolist()
    max_valid_actions = max(n_valid_actions) if n_valid_actions else 0.0
    if max_valid_actions <= 0:
        raise RuntimeError("Rashomon dataset has no valid-action labels.")

    surrogate_threshold = float(max_valid_actions / (1.0 + max_valid_actions))

    actor.eval()
    with torch.no_grad():
        all_obs = rashomon_dataset.tensors[0]
        action_mask = rashomon_dataset.tensors[1]
        logits = actor(all_obs)

        selected_inverse_temp: int | None = None
        min_action_mass = float("-inf")
        for inverse_temp in range(inverse_temp_start, inverse_temp_max + 1):
            probs = torch.softmax(logits * inverse_temp, dim=1)
            valid_action_mass = (probs * action_mask).sum(dim=1)
            min_action_mass = float(valid_action_mass.min().item())
            if min_action_mass >= surrogate_threshold:
                selected_inverse_temp = inverse_temp
                break

        if selected_inverse_temp is None:
            raise ValueError(
                "Could not find inverse temperature satisfying surrogate threshold. "
                f"Best min valid-action mass={min_action_mass:.6f} < threshold={surrogate_threshold:.6f}",
            )

    interval_trainer = IntervalTrainer(
        model=actor,
        min_acc_limit=surrogate_threshold,
        seed=seed,
        n_iters=rashomon_n_iters, # type: ignore
        min_acc_increment=0,
        T=selected_inverse_temp,
        checkpoint=checkpoint, # type: ignore
    )
    # multi_label = (max_valid_actions > 1.0)
    interval_trainer.compute_rashomon_set(
        dataset=rashomon_dataset,
        multi_label=True,
        aggregation=aggregation, # type: ignore
    )

    cert_min_threshold = 1.0
    cert_values = [_certificate_to_float(cert) for cert in interval_trainer.certificates]
    valid_indices = [i for i, cert in enumerate(cert_values) if cert >= cert_min_threshold]
    if not valid_indices:
        best_cert = max(cert_values) if cert_values else float("-inf")
        raise ValueError(
            f"No Rashomon certificate satisfied cert_min_threshold={cert_min_threshold:.3f}. "
            f"Best certificate={best_cert:.6f}",
        )

    selected_idx = valid_indices[-1]
    bounded_model = interval_trainer.bounds[selected_idx]
    param_bounds_l = [p.detach().cpu() for p in bounded_model.param_l]
    param_bounds_u = [p.detach().cpu() for p in bounded_model.param_u]
    return (
        param_bounds_l,
        param_bounds_u,
        bounded_model,
        selected_inverse_temp,
        surrogate_threshold,
        cert_values,
        selected_idx,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downstream adaptation with trajectory Rashomon bounds and PPO-PGD.",
    )
    parser.add_argument("--layout", type=str, default="diagonal_30x30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--activation",
        type=str,
        choices=["tanh", "relu"],
        default="relu",
    )
    parser.add_argument(
        "--source-env-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "source_envs.yaml",
    )
    parser.add_argument(
        "--downstream-env-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "downstream_envs.yaml",
    )
    parser.add_argument(
        "--source-settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "train_source_policy_settings.yaml",
    )
    parser.add_argument(
        "--adapt-settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "downstream_adaptation_settings_ppo.yaml",
        help="Shared downstream settings file with PPO/common per-layout config.",
    )
    parser.add_argument(
        "--rashomon-settings-file",
        type=Path,
        default=Path(__file__).resolve().parent / "settings" / "downstream_adaptation_settings_rashomon.yaml",
        help="Rashomon-specific downstream settings file.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=None,
        help="Optional explicit source checkpoint directory. Defaults to outputs/<layout>/seed_<seed>/source",
    )
    parser.add_argument(
        "--run-subdir",
        type=str,
        default="downstream_rashomon",
        help="Subdirectory under outputs/<layout>/seed_<seed>/ where outputs are saved.",
    )
    parser.add_argument(
        "--disable-task-neutralization",
        action="store_true",
        help="Disable first-layer task-feature neutralization before adaptation.",
    )
    parser.add_argument(
        "--rashomon-n-iters",
        type=int,
        default=None,
        help="Optional override for Rashomon iterations. Defaults to YAML method settings.",
    )
    parser.add_argument(
        "--rashomon-min-hard-spec",
        type=float,
        default=1.0,
        help="Required minimum hard specification certificate level for selecting Rashomon bounds.",
    )
    parser.add_argument(
        "--rashomon-surrogate-aggregation",
        type=str,
        choices=["mean", "min"],
        default=None,
        help="Optional override for surrogate aggregation across behaviour-critical states. Defaults to YAML method settings.",
    )
    parser.add_argument(
        "--inverse-temp-start",
        type=int,
        default=None,
        help="Smallest inverse temperature considered for soft multi-label thresholding.",
    )
    parser.add_argument(
        "--inverse-temp-max",
        type=int,
        default=None,
        help="Largest inverse temperature considered for soft multi-label thresholding.",
    )
    parser.add_argument(
        "--rashomon-checkpoint",
        type=int,
        default=None,
        help="Checkpoint period passed to IntervalTrainer.",
    )
    args = parser.parse_args()

    source_envs = _load_yaml(args.source_env_file)
    downstream_envs = _load_yaml(args.downstream_env_file)
    source_settings = _load_yaml(args.source_settings_file)
    adapt_settings = _load_yaml(args.adapt_settings_file)
    rashomon_settings = _load_yaml(args.rashomon_settings_file)

    if args.layout not in source_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.source_env_file}")
    if args.layout not in downstream_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.downstream_env_file}")
    if args.layout not in source_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.source_settings_file}")
    if args.layout not in adapt_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.adapt_settings_file}")
    if args.layout not in rashomon_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.rashomon_settings_file}")

    source_cfg = source_envs[args.layout]
    downstream_cfg = downstream_envs[args.layout]

    source_settings_cfg = source_settings[args.layout]
    if not isinstance(source_settings_cfg, dict) or "ppo" not in source_settings_cfg:
        raise ValueError(
            f"Layout '{args.layout}' in {args.source_settings_file} does not define a 'ppo' section.",
        )
    source_ppo_cfg = source_settings_cfg["ppo"]

    adapt_cfg = adapt_settings[args.layout]
    if not isinstance(adapt_cfg, dict):
        raise ValueError(
            f"Expected dict config for layout '{args.layout}' in {args.adapt_settings_file}, got {type(adapt_cfg)}.",
        )
    if "ppo" not in adapt_cfg:
        success = adapt_cfg.get("success", None)
        notes = adapt_cfg.get("notes", None)
        available_keys = ", ".join(sorted(str(k) for k in adapt_cfg.keys()))
        fallback_file = _find_layout_with_ppo(args.layout, args.adapt_settings_file)
        fallback_hint = (
            f" A matching fallback exists: --adapt-settings-file {fallback_file}."
            if fallback_file is not None
            else ""
        )
        raise ValueError(
            f"Layout '{args.layout}' in {args.adapt_settings_file} has no 'ppo' section "
            f"(success={success}, keys=[{available_keys}]). "
            f"notes={notes!r}.{fallback_hint}",
        )
    adapt_ppo_cfg = adapt_cfg["ppo"]
    if not isinstance(adapt_ppo_cfg, dict):
        raise ValueError(
            f"Expected 'ppo' to be a dict for layout '{args.layout}' in {args.adapt_settings_file}.",
        )
    rashomon_layout_cfg = rashomon_settings[args.layout]
    if not isinstance(rashomon_layout_cfg, dict):
        raise ValueError(
            f"Expected dict config for layout '{args.layout}' in {args.rashomon_settings_file}, "
            f"got {type(rashomon_layout_cfg)}.",
        )
    if "rashomon" in rashomon_layout_cfg:
        adapt_rashomon_cfg = rashomon_layout_cfg["rashomon"]
    elif any(
        k in rashomon_layout_cfg
        for k in (
            "rashomon_n_iters",
            "rashomon_min_hard_spec",
            "rashomon_surrogate_aggregation",
            "inverse_temp_start",
            "inverse_temp_max",
            "rashomon_checkpoint",
        )
    ):
        # Backward-compatible support for flat Rashomon config blocks.
        adapt_rashomon_cfg = rashomon_layout_cfg
    else:
        adapt_rashomon_cfg = {}
    if not isinstance(adapt_rashomon_cfg, dict):
        raise ValueError(
            f"Expected Rashomon config dict for layout '{args.layout}' in {args.rashomon_settings_file}.",
        )

    rashomon_n_iters = int(
        args.rashomon_n_iters
        if args.rashomon_n_iters is not None
        else adapt_rashomon_cfg.get("rashomon_n_iters", 50_000),
    )
    # rashomon_min_hard_spec = float(
    #     args.rashomon_min_hard_spec
    #     if args.rashomon_min_hard_spec is not None
    #     else adapt_rashomon_cfg.get("rashomon_min_hard_spec", 1.0),
    # )
    # surrogate_aggregation = str(
    #     args.rashomon_surrogate_aggregation
    #     if args.rashomon_surrogate_aggregation is not None
    #     else adapt_rashomon_cfg.get("rashomon_surrogate_aggregation", "min"),
    # )
    rashomon_min_hard_spec = 1.0
    surrogate_aggregation = 'min'
    inverse_temp_start = int(
        args.inverse_temp_start
        if args.inverse_temp_start is not None
        else adapt_rashomon_cfg.get("inverse_temp_start", 10),
    )
    inverse_temp_max = int(
        args.inverse_temp_max
        if args.inverse_temp_max is not None
        else adapt_rashomon_cfg.get("inverse_temp_max", 1000),
    )
    rashomon_checkpoint = int(
        args.rashomon_checkpoint
        if args.rashomon_checkpoint is not None
        else adapt_rashomon_cfg.get("rashomon_checkpoint", 100),
    )

    source_task_num = float(adapt_cfg.get("source_task_num", 0.0))
    downstream_task_num = float(adapt_cfg.get("downstream_task_num", 1.0))
    warm_actor = bool(adapt_cfg.get("warm_start", {}).get("actor", True))
    warm_critic = bool(adapt_cfg.get("warm_start", {}).get("critic", True))
    train_shaped = bool(adapt_cfg.get("train_shaped", False))
    if not warm_actor:
        raise ValueError("This script expects actor warm-start (warm_start.actor=true).")

    source_map: list[str] = source_cfg["env1_map"]
    downstream_map: list[str] = downstream_cfg["env2_map"]
    max_episode_steps = int(source_cfg["max_episode_steps"])
    downstream_max_episode_steps = int(downstream_cfg.get("max_episode_steps", max_episode_steps))
    hidden = int(source_ppo_cfg["hidden"])

    source_run_dir = (
        args.source_run_dir
        if args.source_run_dir is not None
        else args.outputs_root / args.layout / f"seed_{args.seed}" / "source"
    )
    actor_ckpt = source_run_dir / "actor.pt"
    critic_ckpt = source_run_dir / "critic.pt"

    if not actor_ckpt.exists():
        raise FileNotFoundError(f"Source actor checkpoint not found: {actor_ckpt}")
    if warm_critic and not critic_ckpt.exists():
        raise FileNotFoundError(f"Source critic checkpoint not found: {critic_ckpt}")

    source_env_for_dim = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
    )
    obs_dim = int(source_env_for_dim.observation_space.shape[0]) # type: ignore
    n_actions = int(source_env_for_dim.action_space.n) # type: ignore
    source_env_for_dim.close()

    source_actor, source_critic = build_actor_critic(
        obs_dim=obs_dim,
        hidden=hidden,
        activation=args.activation,
    )
    source_actor.load_state_dict(torch.load(actor_ckpt, map_location="cpu"))
    if warm_critic:
        source_critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))
    source_actor_for_dataset = copy.deepcopy(source_actor)

    source_rollout_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
    )
    rashomon_dataset, source_state_action_pairs = create_source_trajectory_rashomon_dataset(
        actor=source_actor_for_dataset,
        env=source_rollout_env,
        seed=args.seed,
        n_actions=n_actions,
    )
    source_rollout_env.close()

    print(
        f"Built source trajectory Rashomon dataset: {len(rashomon_dataset)} samples "
        f"(trajectory steps: {len(source_state_action_pairs)}).",
    )

    (
        param_bounds_l,
        param_bounds_u,
        bounded_model,
        selected_inverse_temp,
        surrogate_threshold,
        cert_values,
        selected_cert_idx,
    ) = compute_rashomon_bounds(
        actor=copy.deepcopy(source_actor),
        rashomon_dataset=rashomon_dataset,
        seed=args.seed,
        rashomon_n_iters=rashomon_n_iters,
        aggregation=surrogate_aggregation,
        inverse_temp_start=inverse_temp_start,
        inverse_temp_max=inverse_temp_max,
        checkpoint=rashomon_checkpoint,
    )

    print(
        f"Rashomon bounds ready: aggregation={surrogate_aggregation} | "
        f"min_hard_spec={rashomon_min_hard_spec:.3f} | selected_cert={cert_values[selected_cert_idx]:.4f} "
        f"(idx={selected_cert_idx}) | inverse_temp={selected_inverse_temp}",
    )

    task_transform_cfg = adapt_cfg.get("pre_adaptation_transform", {})
    do_task_neutralization = (
        bool(task_transform_cfg.get("task_feature_neutralization", False))
        and not args.disable_task_neutralization
    )
    task_feature_index = int(task_transform_cfg.get("task_feature_index", 2))
    if do_task_neutralization:
        neutralize_task_feature(source_actor, task_feature_index, downstream_task_num)
        if warm_critic:
            neutralize_task_feature(source_critic, task_feature_index, downstream_task_num)

    ppo_cfg = PPOConfig(
        seed=int(adapt_ppo_cfg.get("seed", args.seed)),
        total_timesteps=int(adapt_ppo_cfg["total_timesteps"]),
        eval_episodes=int(adapt_ppo_cfg.get("eval_episodes", 1)),
        rollout_steps=int(adapt_ppo_cfg["rollout_steps"]),
        update_epochs=int(adapt_ppo_cfg["update_epochs"]),
        minibatch_size=int(adapt_ppo_cfg["minibatch_size"]),
        gamma=float(adapt_ppo_cfg["gamma"]),
        gae_lambda=float(adapt_ppo_cfg["gae_lambda"]),
        clip_coef=float(adapt_ppo_cfg["clip_coef"]),
        ent_coef=float(adapt_ppo_cfg["ent_coef"]),
        vf_coef=float(adapt_ppo_cfg["vf_coef"]),
        lr=float(adapt_ppo_cfg["lr"]),
        max_grad_norm=float(adapt_ppo_cfg["max_grad_norm"]),
        device=args.device,
        early_stop=bool(adapt_ppo_cfg.get("early_stop", False)),
        early_stop_deterministic_total_reward_threshold=adapt_ppo_cfg.get(
            "early_stop_deterministic_total_reward_threshold",
            None,
        ),
        early_stop_deterministic_eval_episodes=int(
            adapt_ppo_cfg.get("early_stop_deterministic_eval_episodes", 1),
        ),
    )

    print(
        f"Adapting {args.layout} with Rashomon-PGD | source_task={source_task_num} -> downstream_task={downstream_task_num} | "
        f"warm_critic={warm_critic} | task_neutralization={do_task_neutralization} | "
        f"train_shaped={train_shaped}",
    )
    if train_shaped:
        print(
            "Using sparse-reward downstream environment for periodic eval and early-stop checks "
            "(training still uses shaped rewards).",
        )

    train_env = make_env_from_layout(
        downstream_map,
        max_episode_steps,
        task_num=downstream_task_num,
        shaped=train_shaped,
    )
    early_stop_eval_env = make_env_from_layout(
        downstream_map,
        max_episode_steps,
        task_num=downstream_task_num,
        shaped=False,
    )
    actor, critic, training_data = ppo_train(  # type: ignore[assignment]
        train_env,
        ppo_cfg,
        actor_warm_start=source_actor,
        critic_warm_start=(source_critic if warm_critic else None),
        actor_param_bounds_l=param_bounds_l,
        actor_param_bounds_u=param_bounds_u,
        early_stop_eval_env=early_stop_eval_env,
        return_training_data=True,
    )

    eval_episodes = int(adapt_cfg.get("downstream_eval", {}).get("episodes", 1))
    source_eval_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
    )
    source_mean_reward, source_std_reward, source_failure_rate = evaluate(
        source_eval_env,
        actor,
        episodes=eval_episodes,
        deterministic=True,
        device=args.device,
    )
    source_eval_env.close()

    downstream_eval_env = make_env_from_layout(
        downstream_map,
        downstream_max_episode_steps,
        task_num=downstream_task_num,
        shaped=False,
    )
    downstream_mean_reward, downstream_std_reward, downstream_failure_rate = evaluate(
        downstream_eval_env,
        actor,
        episodes=eval_episodes,
        deterministic=True,
        device=args.device,
    )
    downstream_eval_env.close()

    downstream_run_dir = args.outputs_root / args.layout / f"seed_{args.seed}" / args.run_subdir
    downstream_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    source_pairs_path = downstream_run_dir / "rashomon_source_policy_state_action_pairs.yaml"
    rashomon_dataset_path = downstream_run_dir / "rashomon_dataset.pt"
    bounded_model_path = downstream_run_dir / "rashomon_bounded_model.pt"
    bounds_path = downstream_run_dir / "rashomon_param_bounds.pt"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)
    torch.save(rashomon_dataset, rashomon_dataset_path)
    torch.save(bounded_model, bounded_model_path)
    torch.save(
        {
            "param_bounds_l": param_bounds_l,
            "param_bounds_u": param_bounds_u,
        },
        bounds_path,
    )
    source_pairs_path.write_text(
        yaml.safe_dump(source_state_action_pairs, sort_keys=False),
        encoding="utf-8",
    )

    source_render_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
        render_mode="rgb_array",
    )
    try:
        plot_episode(
            env=source_render_env,
            actor=actor,
            seed=args.seed,
            deterministic=True,
            save_path=str(source_plot_path),
            title=f"Rashomon Adapted Policy on Source Task: {args.layout}",
        )
    finally:
        source_render_env.close()

    downstream_render_env = make_env_from_layout(
        downstream_map,
        downstream_max_episode_steps,
        task_num=downstream_task_num,
        shaped=False,
        render_mode="rgb_array",
    )
    try:
        plot_episode(
            env=downstream_render_env,
            actor=actor,
            seed=args.seed,
            deterministic=True,
            save_path=str(downstream_plot_path),
            title=f"Rashomon Adapted Policy on Downstream Task: {args.layout}",
        )
    finally:
        downstream_render_env.close()

    summary = {
        "layout": args.layout,
        "seed": args.seed,
        "source_task_num": source_task_num,
        "downstream_task_num": downstream_task_num,
        "eval_episodes": int(eval_episodes),
        "source_eval_episodes": int(eval_episodes),
        "downstream_eval_episodes": int(eval_episodes),
        "warm_start_actor": warm_actor,
        "warm_start_critic": warm_critic,
        "train_shaped": train_shaped,
        "task_feature_neutralization": do_task_neutralization,
        "task_feature_index": task_feature_index,
        "activation": args.activation,
        "rashomon_dataset_size": int(len(rashomon_dataset)),
        "source_trajectory_steps": int(len(source_state_action_pairs)),
        "rashomon_n_iters": int(rashomon_n_iters),
        "surrogate_aggregation": surrogate_aggregation,
        "inverse_temp_start": int(inverse_temp_start),
        "inverse_temp_max": int(inverse_temp_max),
        "rashomon_checkpoint": int(rashomon_checkpoint),
        "surrogate_threshold": float(surrogate_threshold),
        "inverse_temperature": int(selected_inverse_temp),
        "selected_certificate_index": int(selected_cert_idx),
        "selected_certificate": float(cert_values[selected_cert_idx]),
        "all_certificates": [float(v) for v in cert_values],
        "source_mean_reward": float(source_mean_reward),
        "source_std_reward": float(source_std_reward),
        "source_failure_rate": float(source_failure_rate),
        "downstream_mean_reward": float(downstream_mean_reward),
        "downstream_std_reward": float(downstream_std_reward),
        "downstream_failure_rate": float(downstream_failure_rate),
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
        "rashomon_source_policy_state_action_pairs_path": str(source_pairs_path),
        "rashomon_dataset_path": str(rashomon_dataset_path),
        "rashomon_bounded_model_path": str(bounded_model_path),
        "rashomon_param_bounds_path": str(bounds_path),
        "trajectory_source_plot_path": str(source_plot_path),
        "trajectory_downstream_plot_path": str(downstream_plot_path),
        "source_checkpoint_dir": str(source_run_dir),
    }
    (downstream_run_dir / "run_summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    print(
        f"Source eval ({eval_episodes} ep): mean_reward={source_mean_reward:.3f}, "
        f"std={source_std_reward:.3f}, failure_rate={source_failure_rate:.3f}",
    )
    print(
        f"Downstream eval ({eval_episodes} ep): mean_reward={downstream_mean_reward:.3f}, "
        f"std={downstream_std_reward:.3f}, failure_rate={downstream_failure_rate:.3f}",
    )
    print(f"Saved actor: {actor_path}")
    print(f"Saved critic: {critic_path}")
    print(f"Saved Rashomon dataset: {rashomon_dataset_path}")
    print(f"Saved Rashomon bounded model: {bounded_model_path}")
    print(f"Saved source trajectory plot: {source_plot_path}")
    print(f"Saved downstream trajectory plot: {downstream_plot_path}")


if __name__ == "__main__":
    main()
