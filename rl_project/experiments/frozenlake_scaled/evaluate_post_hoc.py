"""Post-hoc evaluation of saved FrozenLake scaled policies for one layout/seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from rl_project.experiments.frozenlake_scaled.train_source_policy import build_actor_critic, make_env_from_layout
from rl_project.utils.gymnasium_utils import plot_episode
from rl_project.utils.ppo_utils import evaluate

POLICY_KIND_TO_DIR = {
    "source": "source",
    "downstream": "downstream",
    "downstream_ewc": "downstream_ewc",
    "downstream_rashomon": "downstream_rashomon",
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _policy_dir(outputs_root: Path, layout: str, seed: int, kind: str) -> Path:
    return outputs_root / layout / f"seed_{seed}" / kind


def _resolve_actor_path(policy_dir: Path, policy_name: str) -> Path:
    candidates = [
        policy_dir / "actor.pt",
        policy_dir / "unconstrained_actor.pt",
        policy_dir / "ewc_actor.pt",
        policy_dir / "rashomon_actor.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{policy_name.capitalize()} actor checkpoint not found in {policy_dir}. "
        f"Tried: {', '.join(str(p.name) for p in candidates)}",
    )


def _infer_activation(policy_dir: Path, requested_activation: str, fallback: str = "relu") -> str:
    if requested_activation != "auto":
        return requested_activation
    summary_path = policy_dir / "run_summary.yaml"
    if summary_path.exists():
        try:
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = None
        if isinstance(summary, dict):
            activation = summary.get("activation")
            if activation in {"tanh", "relu"}:
                return activation
    return fallback


def _evaluate_actor(
    actor: torch.nn.Sequential,
    env_map: list[str],
    max_episode_steps: int,
    *,
    task_num: float,
    episodes: int,
    device: str,
) -> tuple[float, float, float]:
    env = make_env_from_layout(
        env_map,
        max_episode_steps,
        task_num=task_num,
        shaped=False,
    )
    try:
        return evaluate(
            env,
            actor,
            episodes=episodes,
            deterministic=True,
            device=device,
        )
    finally:
        env.close()


def _plot_trajectory(
    actor: torch.nn.Sequential,
    env_map: list[str],
    max_episode_steps: int,
    *,
    task_num: float,
    seed: int,
    plot_path: Path,
    title: str,
) -> None:
    env = make_env_from_layout(
        env_map,
        max_episode_steps,
        task_num=task_num,
        shaped=False,
        render_mode="rgb_array",
    )
    try:
        plot_episode(
            env=env,
            actor=actor,
            seed=seed,
            deterministic=True,
            save_path=str(plot_path),
            title=title,
        )
    finally:
        env.close()


def _seed_from_dir_name(name: str) -> int | None:
    if not name.startswith("seed_"):
        return None
    suffix = name.removeprefix("seed_")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance ** 0.5


def evaluate_all_policies_across_seeds(
    *,
    outputs_root: Path,
    source_envs: dict,
    downstream_envs: dict,
    source_settings: dict,
    adapt_settings: dict,
    episodes: int,
    device: str,
    activation: str,
    layout_filter: str | None = None,
    seed_filter: list[int] | None = None,
    policy_kinds: list[str] | None = None,
) -> dict:
    selected_policy_kinds = policy_kinds or list(POLICY_KIND_TO_DIR.keys())
    for policy_kind in selected_policy_kinds:
        if policy_kind not in POLICY_KIND_TO_DIR:
            raise ValueError(
                f"Unknown policy kind '{policy_kind}'. Valid options: {sorted(POLICY_KIND_TO_DIR)}",
            )

    if layout_filter is None:
        layout_dirs = sorted([path for path in outputs_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    else:
        layout_path = outputs_root / layout_filter
        if not layout_path.exists():
            raise FileNotFoundError(f"Layout directory not found in outputs: {layout_path}")
        layout_dirs = [layout_path]

    seed_filter_set = set(seed_filter) if seed_filter is not None else None
    layout_summaries: list[dict] = []
    global_values_by_pair: dict[tuple[str, str], list[float]] = {}

    for layout_dir in layout_dirs:
        layout = layout_dir.name
        if layout not in source_envs or layout not in downstream_envs or layout not in source_settings:
            print(f"[skip] layout={layout} missing env/settings metadata.")
            continue

        seed_dirs: list[tuple[int, Path]] = []
        for child in layout_dir.iterdir():
            if not child.is_dir():
                continue
            seed = _seed_from_dir_name(child.name)
            if seed is None:
                continue
            if seed_filter_set is not None and seed not in seed_filter_set:
                continue
            seed_dirs.append((seed, child))
        seed_dirs.sort(key=lambda item: item[0])

        if not seed_dirs:
            print(f"[skip] layout={layout} has no matching seed directories.")
            continue

        source_cfg = source_envs[layout]
        downstream_cfg = downstream_envs[layout]
        source_ppo_cfg = source_settings[layout]["ppo"]
        adapt_cfg = adapt_settings.get(layout, {})

        source_map: list[str] = source_cfg["env1_map"]
        downstream_map: list[str] = downstream_cfg["env2_map"]
        max_episode_steps = int(source_cfg["max_episode_steps"])
        hidden = int(source_ppo_cfg["hidden"])
        source_task_num = float(adapt_cfg.get("source_task_num", 0.0))
        downstream_task_num = float(adapt_cfg.get("downstream_task_num", 1.0))

        obs_env = make_env_from_layout(
            source_map,
            max_episode_steps,
            task_num=source_task_num,
            shaped=False,
        )
        obs_dim = int(obs_env.observation_space.shape[0])
        obs_env.close()

        env_specs = {
            "source": {"map": source_map, "task_num": source_task_num},
            "downstream": {"map": downstream_map, "task_num": downstream_task_num},
        }

        seed_level_results: list[dict] = []
        values_by_pair: dict[tuple[str, str], list[float]] = {}
        seeds_by_pair: dict[tuple[str, str], set[int]] = {}

        for seed, seed_dir in seed_dirs:
            for policy_kind in selected_policy_kinds:
                policy_subdir = POLICY_KIND_TO_DIR[policy_kind]
                policy_dir = seed_dir / policy_subdir
                if not policy_dir.exists():
                    continue
                try:
                    actor_path = _resolve_actor_path(policy_dir, policy_kind)
                except FileNotFoundError:
                    continue

                policy_activation = _infer_activation(policy_dir, activation)
                actor, _ = build_actor_critic(
                    obs_dim=obs_dim,
                    hidden=hidden,
                    activation=policy_activation,
                )
                try:
                    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
                except Exception as exc:
                    print(
                        f"[skip] layout={layout} seed={seed} policy={policy_kind} "
                        f"failed to load actor from {actor_path}: {exc}",
                    )
                    continue
                actor.to(device)

                for env_name, env_spec in env_specs.items():
                    mean_reward, std_reward, failure_rate = _evaluate_actor(
                        actor=actor,
                        env_map=env_spec["map"],
                        max_episode_steps=max_episode_steps,
                        task_num=float(env_spec["task_num"]),
                        episodes=episodes,
                        device=device,
                    )
                    result_row = {
                        "seed": seed,
                        "policy": policy_kind,
                        "environment": env_name,
                        "mean_reward": float(mean_reward),
                        "std_reward": float(std_reward),
                        "failure_rate": float(failure_rate),
                        "actor_path": str(actor_path),
                        "activation": policy_activation,
                    }
                    seed_level_results.append(result_row)
                    key = (policy_kind, env_name)
                    values_by_pair.setdefault(key, []).append(float(mean_reward))
                    seeds_by_pair.setdefault(key, set()).add(seed)
                    global_values_by_pair.setdefault(key, []).append(float(mean_reward))

        policy_environment_stats: list[dict] = []
        for policy_kind in selected_policy_kinds:
            for env_name in ("source", "downstream"):
                key = (policy_kind, env_name)
                values = values_by_pair.get(key, [])
                if not values:
                    continue
                mean_reward, std_reward = _mean_std(values)
                policy_environment_stats.append(
                    {
                        "policy": policy_kind,
                        "environment": env_name,
                        "num_seed_runs": len(seeds_by_pair[key]),
                        "mean_total_reward_across_seeds": float(mean_reward),
                        "std_total_reward_across_seeds": float(std_reward),
                    },
                )

        if not policy_environment_stats:
            print(f"[skip] layout={layout} found no evaluable policies across selected seeds.")
            continue

        print(f"\nLayout={layout} | seeds={len(seed_dirs)} | episodes_per_eval={episodes}")
        for row in policy_environment_stats:
            print(
                f"  {row['policy']:18s} on {row['environment']:10s} | "
                f"n={row['num_seed_runs']:2d} mean={row['mean_total_reward_across_seeds']:.3f} "
                f"std={row['std_total_reward_across_seeds']:.3f}",
            )

        layout_summaries.append(
            {
                "layout": layout,
                "requested_seeds": [seed for seed, _ in seed_dirs],
                "source_task_num": source_task_num,
                "downstream_task_num": downstream_task_num,
                "policy_environment_stats": policy_environment_stats,
                "seed_level_results": seed_level_results,
            },
        )

    global_policy_environment_stats: list[dict] = []
    for policy_kind in selected_policy_kinds:
        for env_name in ("source", "downstream"):
            key = (policy_kind, env_name)
            values = global_values_by_pair.get(key, [])
            if not values:
                continue
            mean_reward, std_reward = _mean_std(values)
            global_policy_environment_stats.append(
                {
                    "policy": policy_kind,
                    "environment": env_name,
                    "num_layout_seed_evals": len(values),
                    "mean_total_reward": float(mean_reward),
                    "std_total_reward": float(std_reward),
                },
            )

    return {
        "outputs_root": str(outputs_root),
        "layout_filter": layout_filter,
        "seed_filter": list(seed_filter) if seed_filter is not None else None,
        "episodes": episodes,
        "device": device,
        "activation": activation,
        "policy_kinds": selected_policy_kinds,
        "layouts": layout_summaries,
        "global_policy_environment_stats": global_policy_environment_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved source/downstream policies for a given layout and seed.",
    )
    parser.add_argument("--layout", type=str, default="diagonal_30x30")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed filter for aggregate mode (used with --aggregate-across-seeds).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--activation",
        type=str,
        choices=["tanh", "relu", "auto"],
        default="auto",
        help="Activation used to construct networks before loading checkpoints (default: auto from run_summary.yaml).",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--aggregate-across-seeds",
        action="store_true",
        help=(
            "Evaluate all selected policies across available seeds for each layout "
            "(set --layout all to scan all layout directories)."
        ),
    )
    parser.add_argument(
        "--policy-kinds",
        type=str,
        nargs="+",
        choices=list(POLICY_KIND_TO_DIR.keys()),
        default=list(POLICY_KIND_TO_DIR.keys()),
        help=(
            "Policy directories to include in aggregate mode. "
            "Choices: source downstream downstream_ewc downstream_rashomon."
        ),
    )
    parser.add_argument(
        "--policies",
        type=str,
        choices=["source", "downstream", "both"],
        default="both",
        help="Which saved policies to evaluate.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["matching", "both_envs"],
        default="both_envs",
        help=(
            "matching: evaluate source policy on source env and downstream policy on downstream env. "
            "both_envs: evaluate each selected policy on both source and downstream envs."
        ),
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
        help="Optional override for source run directory.",
    )
    parser.add_argument(
        "--downstream-run-dir",
        type=Path,
        default=None,
        help="Optional override for downstream run directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit output yaml path. Defaults to outputs/<layout>/seed_<seed>/post_hoc_eval.yaml",
    )
    parser.add_argument(
        "--aggregate-output-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit output yaml path for aggregate mode. "
            "Defaults to outputs/post_hoc_eval_aggregate.yaml (or outputs/<layout>/post_hoc_eval_aggregate.yaml)."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="If set, do not generate trajectory figures.",
    )
    args = parser.parse_args()

    source_envs = _load_yaml(args.source_env_file)
    downstream_envs = _load_yaml(args.downstream_env_file)
    source_settings = _load_yaml(args.source_settings_file)
    adapt_settings = _load_yaml(args.adapt_settings_file)

    if args.aggregate_across_seeds:
        layout_filter = None if args.layout == "all" else args.layout
        summary = evaluate_all_policies_across_seeds(
            outputs_root=args.outputs_root,
            source_envs=source_envs,
            downstream_envs=downstream_envs,
            source_settings=source_settings,
            adapt_settings=adapt_settings,
            episodes=args.episodes,
            device=args.device,
            activation=args.activation,
            layout_filter=layout_filter,
            seed_filter=args.seeds,
            policy_kinds=args.policy_kinds,
        )
        aggregate_output_file = args.aggregate_output_file
        if aggregate_output_file is None:
            if layout_filter is None:
                aggregate_output_file = args.outputs_root / "post_hoc_eval_aggregate.yaml"
            else:
                aggregate_output_file = args.outputs_root / layout_filter / "post_hoc_eval_aggregate.yaml"
        aggregate_output_file.parent.mkdir(parents=True, exist_ok=True)
        aggregate_output_file.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        print(f"Saved aggregate post-hoc evaluation: {aggregate_output_file}")
        return

    if args.layout not in source_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.source_env_file}")
    if args.layout not in downstream_envs:
        raise ValueError(f"Layout '{args.layout}' not found in {args.downstream_env_file}")
    if args.layout not in source_settings:
        raise ValueError(f"Layout '{args.layout}' not found in {args.source_settings_file}")

    source_cfg = source_envs[args.layout]
    downstream_cfg = downstream_envs[args.layout]
    source_ppo_cfg = source_settings[args.layout]["ppo"]
    adapt_cfg = adapt_settings.get(args.layout, {})

    source_map: list[str] = source_cfg["env1_map"]
    downstream_map: list[str] = downstream_cfg["env2_map"]
    max_episode_steps = int(source_cfg["max_episode_steps"])
    hidden = int(source_ppo_cfg["hidden"])
    source_task_num = float(adapt_cfg.get("source_task_num", 0.0))
    downstream_task_num = float(adapt_cfg.get("downstream_task_num", 1.0))

    source_run_dir = args.source_run_dir or _policy_dir(args.outputs_root, args.layout, args.seed, "source")
    downstream_run_dir = args.downstream_run_dir or _policy_dir(args.outputs_root, args.layout, args.seed, "downstream")

    selected_policies = ["source", "downstream"] if args.policies == "both" else [args.policies]
    source_actor_path = _resolve_actor_path(source_run_dir, "source") if "source" in selected_policies else None
    downstream_actor_path = (
        _resolve_actor_path(downstream_run_dir, "downstream") if "downstream" in selected_policies else None
    )

    eval_pairs: list[tuple[str, str]] = []
    if args.eval_mode == "matching":
        for policy_name in selected_policies:
            env_name = "source" if policy_name == "source" else "downstream"
            eval_pairs.append((policy_name, env_name))
    else:
        for policy_name in selected_policies:
            eval_pairs.append((policy_name, "source"))
            eval_pairs.append((policy_name, "downstream"))

    obs_env = make_env_from_layout(
        source_map,
        max_episode_steps,
        task_num=source_task_num,
        shaped=False,
    )
    obs_dim = int(obs_env.observation_space.shape[0])
    obs_env.close()

    actors: dict[str, torch.nn.Sequential] = {}
    actor_activations: dict[str, str] = {}
    if "source" in selected_policies:
        assert source_actor_path is not None
        source_activation = _infer_activation(source_run_dir, args.activation)
        source_actor, _ = build_actor_critic(obs_dim=obs_dim, hidden=hidden, activation=source_activation)
        source_actor.load_state_dict(torch.load(source_actor_path, map_location="cpu"))
        source_actor.to(args.device)
        actors["source"] = source_actor
        actor_activations["source"] = source_activation
        print(f"Loaded source actor: {source_actor_path} (activation={source_activation})")
    if "downstream" in selected_policies:
        assert downstream_actor_path is not None
        downstream_activation = _infer_activation(downstream_run_dir, args.activation)
        downstream_actor, _ = build_actor_critic(
            obs_dim=obs_dim,
            hidden=hidden,
            activation=downstream_activation,
        )
        downstream_actor.load_state_dict(torch.load(downstream_actor_path, map_location="cpu"))
        downstream_actor.to(args.device)
        actors["downstream"] = downstream_actor
        actor_activations["downstream"] = downstream_activation
        print(f"Loaded downstream actor: {downstream_actor_path} (activation={downstream_activation})")

    env_specs = {
        "source": {"map": source_map, "task_num": source_task_num},
        "downstream": {"map": downstream_map, "task_num": downstream_task_num},
    }

    output_file = args.output_file
    if output_file is None:
        output_file = args.outputs_root / args.layout / f"seed_{args.seed}" / "post_hoc_eval.yaml"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = output_file.parent / "post_hoc_plots"
    if not args.skip_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    evaluations: list[dict] = []
    for policy_name, env_name in eval_pairs:
        actor = actors[policy_name]
        env_spec = env_specs[env_name]
        mean_reward, std_reward, failure_rate = _evaluate_actor(
            actor=actor,
            env_map=env_spec["map"],
            max_episode_steps=max_episode_steps,
            task_num=float(env_spec["task_num"]),
            episodes=args.episodes,
            device=args.device,
        )
        trajectory_plot_path = None
        if not args.skip_plots:
            trajectory_plot_path = plot_dir / f"{policy_name}_policy_on_{env_name}_env.png"
            _plot_trajectory(
                actor=actor,
                env_map=env_spec["map"],
                max_episode_steps=max_episode_steps,
                task_num=float(env_spec["task_num"]),
                seed=args.seed,
                plot_path=trajectory_plot_path,
                title=f"{policy_name.capitalize()} Policy on {env_name.capitalize()} Env",
            )
        row = {
            "policy": policy_name,
            "environment": env_name,
            "episodes": args.episodes,
            "deterministic": True,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "failure_rate": float(failure_rate),
            "trajectory_plot_path": str(trajectory_plot_path) if trajectory_plot_path is not None else None,
        }
        evaluations.append(row)
        print(
            f"{policy_name:10s} on {env_name:10s} | "
            f"mean_reward={mean_reward:.3f} std={std_reward:.3f} failure_rate={failure_rate:.3f}",
        )

    summary = {
        "layout": args.layout,
        "seed": args.seed,
        "device": args.device,
        "activation": args.activation,
        "loaded_actor_activations": actor_activations,
        "episodes": args.episodes,
        "policies": selected_policies,
        "eval_mode": args.eval_mode,
        "source_actor_path": str(source_actor_path) if source_actor_path is not None else None,
        "downstream_actor_path": str(downstream_actor_path) if downstream_actor_path is not None else None,
        "source_task_num": source_task_num,
        "downstream_task_num": downstream_task_num,
        "plot_dir": str(plot_dir) if not args.skip_plots else None,
        "evaluations": evaluations,
    }
    output_file.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"Saved post-hoc evaluation: {output_file}")


if __name__ == "__main__":
    main()
