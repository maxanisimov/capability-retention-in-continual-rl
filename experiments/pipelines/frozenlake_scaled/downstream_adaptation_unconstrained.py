"""Train downstream adaptation for scaled FrozenLake layouts with PPO warm-start."""

from __future__ import annotations

import argparse
from pathlib import Path
import os
os.environ["SDL_AUDIODRIVER"] = "dummy" # to disable ALSA warnings when running on headless servers without audio devices
import torch
import yaml

from experiments.pipelines.frozenlake_scaled.train_source_policy import build_actor_critic, make_env_from_layout
from experiments.utils.gymnasium_utils import plot_episode
from experiments.utils.ppo_utils import PPOConfig, evaluate, ppo_train


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
    """Neutralize task-id contribution in the first linear layer for target task value.

    For first-layer pre-activation: z = Wx + b, with x_task = target_task_value,
    this applies:
        b <- b - W_task * target_task_value
        W_task <- 0
    so the task feature no longer shifts activations at adaptation start.
    """
    first = model[0]
    if not isinstance(first, torch.nn.Linear):
        raise ValueError("Expected first layer to be torch.nn.Linear for task-feature neutralization.")

    with torch.no_grad():
        w_task = first.weight[:, task_feature_index].clone()
        first.bias[:] = first.bias - w_task * target_task_value
        first.weight[:, task_feature_index] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream adaptation with PPO for one layout.")
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
        "--disable-task-neutralization",
        action="store_true",
        help="Disable first-layer task-feature neutralization before adaptation.",
    )
    parser.add_argument(
        "--total-timesteps-override",
        type=int,
        default=None,
        help="Optional override for PPO total timesteps.",
    )
    args = parser.parse_args()

    source_envs = _load_yaml(args.source_env_file)
    downstream_envs = _load_yaml(args.downstream_env_file)
    source_settings = _load_yaml(args.source_settings_file)
    adapt_settings = _load_yaml(args.adapt_settings_file)

    if args.layout not in source_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.source_env_file}")
    if args.layout not in downstream_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.downstream_env_file}")
    if args.layout not in source_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.source_settings_file}")
    if args.layout not in adapt_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.adapt_settings_file}")

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

    source_task_num = float(adapt_cfg.get("source_task_num", 0.0))
    downstream_task_num = float(adapt_cfg.get("downstream_task_num", 1.0))
    warm_actor = bool(adapt_cfg.get("warm_start", {}).get("actor", True))
    warm_critic = bool(adapt_cfg.get("warm_start", {}).get("critic", False))
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
    source_env_for_dim.close()

    source_actor, source_critic = build_actor_critic(
        obs_dim=obs_dim,
        hidden=hidden,
        activation=args.activation,
    )
    source_actor.load_state_dict(torch.load(actor_ckpt, map_location="cpu"))
    if warm_critic:
        source_critic.load_state_dict(torch.load(critic_ckpt, map_location="cpu"))

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

    total_timesteps = (
        int(args.total_timesteps_override)
        if args.total_timesteps_override is not None
        else int(adapt_ppo_cfg["total_timesteps"])
    )

    ppo_cfg = PPOConfig(
        seed=int(adapt_ppo_cfg.get("seed", args.seed)),
        total_timesteps=total_timesteps,
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
        f"Adapting {args.layout} | source_task={source_task_num} -> downstream_task={downstream_task_num} | "
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
    actor, critic, training_data = ppo_train( # type: ignore
        train_env,
        ppo_cfg,
        actor_warm_start=source_actor,
        critic_warm_start=(source_critic if warm_critic else None),
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

    downstream_run_dir = args.outputs_root / args.layout / f"seed_{args.seed}" / "downstream_unconstrained"
    downstream_run_dir.mkdir(parents=True, exist_ok=True)

    actor_path = downstream_run_dir / "actor.pt"
    critic_path = downstream_run_dir / "critic.pt"
    training_data_path = downstream_run_dir / "training_data.pt"
    source_plot_path = downstream_run_dir / "trajectory_source.png"
    downstream_plot_path = downstream_run_dir / "trajectory_downstream.png"

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    torch.save(training_data, training_data_path)

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
            title=f"Adapted Policy on Source Task: {args.layout}",
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
            title=f"Adapted Policy on Downstream Task: {args.layout}",
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
        "source_mean_reward": float(source_mean_reward),
        "source_std_reward": float(source_std_reward),
        "source_failure_rate": float(source_failure_rate),
        "downstream_mean_reward": float(downstream_mean_reward),
        "downstream_std_reward": float(downstream_std_reward),
        "downstream_failure_rate": float(downstream_failure_rate),
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "training_data_path": str(training_data_path),
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
    print(f"Saved source trajectory plot: {source_plot_path}")
    print(f"Saved downstream trajectory plot: {downstream_plot_path}")


if __name__ == "__main__":
    main()
